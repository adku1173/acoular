# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements blockwise processing methods in the frequency domain.

.. autosummary::
    :toctree: generated/

    FreqGenerator
    FreqInOut
    RFFT
    IRFFT

"""

import multiprocessing

import numpy as np
from scipy import fft
from traits.api import CLong, Delegate, Either, Float, HasPrivateTraits, Int, Property, Trait, cached_property

from .fastFuncs import calcCSM
from .internal import digest
from .tprocess import SamplesGenerator, TimeInOut

CPU_COUNT = multiprocessing.cpu_count()


class FreqGenerator(HasPrivateTraits):
    """Base class for any generating signal processing block in frequency domain.

    It provides a common interface for all FreqGenerator classes, which
    generate an output via the generator :meth:`result`.
    This class has no real functionality on its own and should not be
    used directly.
    """

    #: Sampling frequency of the signal, defaults to 1.0
    sample_freq = Float(1.0, desc='sampling frequency')

    #: Number of channels
    numchannels = CLong

    #: Number of samples
    numsamples = CLong

    # internal identifier
    digest = Property(depends_on=['sample_freq', 'numchannels', 'numsamples'])

    def _get_digest(self):
        return digest(self)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block)

        Yields
        ------
        No output since `SamplesGenerator` only represents a base class to derive
        other classes from.
        """


class FreqInOut(FreqGenerator):
    """Base class for any frequency domain signal processing block,
    gets a number pf frequencies from :attr:`source` and generates output via the
    generator :meth:`result`.
    """

    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Trait(FreqGenerator)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of channels in output, as given by :attr:`source`.
    numchannels = Delegate('source')

    #: Number of samples in output, as given by :attr:`source`.
    numsamples = Delegate('source')

    # internal identifier
    digest = Property(depends_on=['source.digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def __gt__(self, receiver):
        if isinstance(receiver, (FreqInOut)):
            receiver.source = self
            return receiver
        msg = f'Receiving object {receiver.__class__} must be derived from TimeInOut or FreqInOut.'
        raise TypeError(msg)

    def __lt__(self, source):
        self.source = source
        return self

    def result(self, num):
        """Python generator: dummy function, just echoes the output of source.

        Yields
        ------
        numpy.ndarray
            blocks of shape (num, :attr:`numchannels`),
            whereby num is the number of frequencies.
        """
        yield from self.source.result(num)


class RFFT(FreqInOut):
    """Provides the Fast Fourier Transform (FFT) of multichannel time data."""

    source = Trait(SamplesGenerator)

    # internal identifier
    digest = Property(depends_on=['source.digest'])

    #: Number of workers to use for the FFT calculation
    workers = Int(CPU_COUNT, desc='number of workers to use')

    #: TODO: should implement different normalization methods
    norm = Either(None, 'amplitude')

    def get_blocksize(self, numfreq):
        return (numfreq - 1) * 2 if numfreq % 2 != 0 else numfreq * 2 - 1

    def fftfreq(self, numfreq):
        """Return the Discrete Fourier Transform sample frequencies.

        Returns
        -------
        f : numpy.ndarray
            Array of length :code:`numfreq` containing the sample frequencies.

        """
        blocksize = self.get_blocksize(numfreq)
        return abs(fft.fftfreq(blocksize, 1.0 / self.sample_freq)[: int(blocksize / 2 + 1)])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num):
        """Python generator that yields the FFT spectra block-wise.

        Applies zero padding to the input data if the last returned block
        is shorter than the requested block size.

        Parameters
        ----------
        num : integer
            This parameter defines the number of frequencies to be yielded
            per generator call.

        Yields
        ------
        numpy.ndarray
            FFT spectra of shape (num, :attr:`numchannels`),
            whereby num is the number of frequencies.
        """
        blocksize = self.get_blocksize(num)
        weight = 1 / blocksize if self.norm == 'amplitude' else 1.0
        for data in self.source.result(blocksize):
            # should use additional "out" parameter in the future to avoid reallocation (numpy > 2.0)
            rfft = fft.rfft(data, n=blocksize, axis=0, workers=self.workers) * weight
            rfft[1:-1] *= np.sqrt(2)  # one-sided spectrum correction
            yield rfft


class IRFFT(TimeInOut):
    source = Trait(FreqInOut)

    #: Number of workers to use for the IFFT calculation
    workers = Int(CPU_COUNT, desc='number of workers to use')

    def _validate_num(self, num):
        if num % 2 != 0:
            msg = 'Number of samples must be even'
            raise ValueError(msg)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Allows only even numbers.

        Yields
        ------
        numpy.ndarray
            Yields blocks of shape (num, numchannels).
        """
        # should use additional "out" parameter in the future to avoid reallocation (numpy > 2.0)
        numfreq = int(num / 2 + 1)
        for temp in self.source.result(numfreq):
            yield fft.irfft(temp, n=num, axis=0, workers=self.workers)


class CrossPowerSpectra(FreqInOut):
    #: Data source; :class:`~acoular.fprocess.FreqInOut` or derived object.
    source = Trait(FreqInOut)

    #: The floating-number-precision of entries of csm, eigenvalues and
    #: eigenvectors, corresponding to numpy dtypes. Default is 64 bit.
    precision = Trait('complex128', 'complex64', desc='precision of the fft')

    #: Calculation mode, either 'full' or 'upper'.
    #: 'full' calculates the full cross-spectral matrix, 'upper' calculates
    # only the upper triangle. Default is 'full'.
    calc_mode = Trait('full', 'upper', desc='calculation mode')

    #: Number of channels in output, as given by :attr:`source`.
    numchannels = Property(depends_on='source.numchannels')

    #: Normalization method, either None or 'psd' (Power Spectral Density).
    norm = Either(None, 'psd')

    # internal identifier
    digest = Property(depends_on=['source.digest', 'calc_mode', 'norm'])

    @cached_property
    def _get_numchannels(self):
        n = self.source.numchannels
        return n**2 if self.calc_mode == 'full' else n + n * (n - 1) / 2

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Yields
        ------
        numpy.ndarray
            Yields blocks of shape (num, numchannels).
        """
        numspec = self.numchannels
        blocksize = ((num - 1) * 2) ** 2
        weight = 1.0 / blocksize
        if self.norm == 'psd':
            weight *= blocksize / self.sample_freq
        for data in self.source.result(num):
            csm_upper = np.zeros((num, self.source.numchannels, self.source.numchannels), dtype=self.precision)
            calcCSM(csm_upper, data)  # TODO: requires new method (only temporary solution) # noqa: TD002, TD003, FIX002
            if self.calc_mode == 'full':
                csm_lower = csm_upper.conj().transpose(0, 2, 1)
                [np.fill_diagonal(csm_lower[cntFreq, :, :], 0) for cntFreq in range(csm_lower.shape[0])]
                yield (csm_lower + csm_upper).reshape(num, -1) * weight
            else:
                yield csm_upper.reshape(num, -1)[:, :numspec] * weight