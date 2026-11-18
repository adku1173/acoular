# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Frequency-dependent compression of computational grids.

.. inheritance-diagram:: acoular.compression.CompressedGrid
    :top-classes: acoular.grids.Grid
    :parts: 1

.. autosummary::
    :toctree: generated/

    CompressedGrid
    CG2
    DRAE
    DRPE
"""

from abc import abstractmethod

from .base import AcoularBase
from .fbeamform import BeamformerBase, SteeringVector
from .grids import Grid
from .spectra import PowerSpectra

import numpy as np
from traits.api import Float, Instance, Property, Range, property_depends_on


class CompressedGrid(Grid):
    """Abstract grid selecting a frequency-dependent subset of an original grid.

    Subclasses implement :meth:`_calc_mask` to return a one-dimensional Boolean mask with one
    entry per point in :attr:`grid`. The base class validates and caches this mask, preserves
    the original point order, and derives the compressed geometry. Empty selections raise
    :class:`ValueError`.

    Subclasses must include every input to their selection method in the :attr:`digest`
    dependencies. The original grid and frequency must remain dependencies, directly or through
    another input's digest. Declare the extended digest with ``fget=AcoularBase._get_digest``.
    This invalidates the cached selection and geometry whenever the original grid, frequency,
    or method inputs change.

    Notes
    -----
    Set :attr:`freq` explicitly before using this grid for a beamformer frequency line. Existing
    beamformers do not synchronize it automatically. Their results refer to the compressed grid;
    :attr:`indices` provides the mapping back to the original grid. Treat the returned arrays as
    read-only. Use :class:`~acoular.grids.Sector` objects for integration: ``indices`` is an index
    array, not a coordinate-lookup method as on rectangular grids.
    """

    #: Original :class:`~acoular.grids.Grid` from which points are retained.
    grid = Instance(Grid)

    #: Frequency in Hz for which to select grid points. Defaults to 1.0.
    freq = Float(1.0)

    #: Boolean retention mask of shape ``(grid.size,)``. (read-only)
    mask = Property()

    #: Strictly increasing indices of retained points in the original grid. (read-only)
    indices = Property()

    #: A unique identifier including the original grid and current frequency. (read-only)
    digest = Property(fget=AcoularBase._get_digest, depends_on=['grid.digest', 'freq'])

    @abstractmethod
    def _calc_mask(self):
        """Calculate which original grid points to retain at :attr:`freq`.

        Returns
        -------
        numpy.ndarray of bool
            One-dimensional mask of shape ``(grid.size,)``. At least one entry must be ``True``.
            The base class handles validation and caching; subclasses only calculate the mask.
        """

    @property_depends_on(['digest'])
    def _get_mask(self):
        mask = np.asarray(self._calc_mask())
        if mask.dtype != np.dtype(bool):
            msg = 'The compression mask must be Boolean.'
            raise ValueError(msg)
        if mask.shape != (self.grid.size,):
            msg = f'The compression mask must have shape ({self.grid.size},), got {mask.shape}.'
            raise ValueError(msg)
        if not mask.any():
            msg = 'Compression must retain at least one grid point.'
            raise ValueError(msg)
        return mask

    @property_depends_on(['digest'])
    def _get_indices(self):
        return np.flatnonzero(self.mask)

    @property_depends_on(['digest'])
    def _get_pos(self):
        return self.grid.pos[:, self.indices]

    @property_depends_on(['digest'])
    def _get_size(self):
        return self.indices.size

    @property_depends_on(['digest'])
    def _get_shape(self):
        return (self.size,)


class CG2(CompressedGrid):
    """Compress a grid by retaining points above a diagonal-removed DAS threshold.

    Implements the grid-selection step of DAMAS-CG2 described by Ma and Liu (2017),
    https://doi.org/10.1016/j.jsv.2017.03.027. A conventional delay-and-sum beamformer is evaluated
    on the original grid with CSM diagonal removal. Points are retained only where its output
    is strictly greater than :attr:`threshold`. The default threshold of zero reproduces the
    positive-beamforming criterion; higher thresholds allow more aggressive compression.

    The original :attr:`grid` is a read-only alias of ``steer.grid``. The supplied steering vector
    must describe the original grid, not this compressed grid. It and the spectral data are not
    modified by compression. Any downstream beamformer must use a separate steering vector whose
    grid is this object.

    Frequency selection follows :meth:`~acoular.fbeamform.BeamformerBase.synthetic`. Set
    :attr:`freq` to a resolved FFT frequency to avoid frequency-substitution warnings.
    """

    #: Steering vector describing the original, uncompressed grid.
    steer = Instance(SteeringVector, args=())

    #: Spectral data used for the preliminary delay-and-sum beamforming map.
    freq_data = Instance(PowerSpectra)

    #: Original grid supplied by :attr:`steer`. (read-only)
    grid = Property()

    #: Nonnegative absolute beamforming threshold in Pa squared. Defaults to zero.
    threshold = Range(low=0.0, value=0.0)

    #: A unique identifier including all inputs to the selection. (read-only)
    digest = Property(
        fget=AcoularBase._get_digest,
        depends_on=['steer.digest', 'freq_data.digest', 'freq', 'threshold'],
    )

    @property_depends_on(['steer.grid'])
    def _get_grid(self):
        return self.steer.grid

    def _calc_mask(self):
        beamformer = BeamformerBase(freq_data=self.freq_data, steer=self.steer, r_diag=True, cached=False)
        return beamformer.synthetic(self.freq).ravel() > self.threshold


class DRAE(CompressedGrid):
    """Reduce a dictionary using all signal and noise eigenvectors (DR-AE).

    Implements Feng et al. (2026), Eq. (19), https://doi.org/10.1121/10.0046883.
    At a single resolved FFT frequency, the full cross-spectral matrix (CSM) supplies
    the covariance eigenvectors. The :attr:`n` largest-eigenvalue eigenvectors span
    the signal subspace; the remaining vectors span the noise subspace. A grid point
    is retained if its transfer vector has strictly more squared projection energy
    in the signal subspace than in the noise subspace.

    Set :attr:`n` explicitly to the assumed number of sources. Its default of zero
    is an unset sentinel, not an estimated source count. Valid selections require
    ``1 <= n < steer.mics.num_mics``. Set :attr:`freq` to an exact frequency returned
    by ``freq_data.fftfreq()``; unresolved frequencies raise :class:`ValueError`.
    No frequency-band averaging or CSM diagonal removal is performed.

    The original :attr:`grid` is a read-only alias of ``steer.grid``. Dictionary
    atoms come from :meth:`~acoular.fbeamform.SteeringVector.transfer`, independently
    of the beamforming weight formulation. Supply a steering vector on the original
    grid and use a separate steering vector on this compressed grid downstream.
    Input geometry and spectral data are not modified.

    Notes
    -----
    The subspace model assumes a full-rank source covariance, sufficient snapshots,
    and spatially white, equal-power sensor noise uncorrelated with the sources.
    Fully coherent sources require separate decorrelation preprocessing; colored
    noise requires an appropriate noise model or consistent whitening of both data
    and dictionary. Finite-sample errors and off-grid sources can cause true-source
    neighborhoods to be discarded. Near-field transfer vectors extend the paper's
    demonstrated far-field ULA formulation without changing the projection rule.
    """

    #: Steering vector describing the original, uncompressed dictionary.
    steer = Instance(SteeringVector, args=())

    #: Full CSM data, either FFT-estimated or imported.
    freq_data = Instance(PowerSpectra)

    #: Original grid supplied by :attr:`steer`. (read-only)
    grid = Property()

    #: Assumed source count. Zero means unset; set explicitly before evaluating a mask.
    n = Range(low=0, value=0)

    #: A unique identifier including all dictionary-reduction inputs. (read-only)
    digest = Property(
        fget=AcoularBase._get_digest,
        depends_on=['steer.digest', 'freq_data.digest', 'freq_data.frequencies', 'freq', 'n'],
    )

    @property_depends_on(['steer.grid'])
    def _get_grid(self):
        return self.steer.grid

    def _calc_mask(self):
        if self.n == 0:
            msg = 'Set n explicitly to the assumed number of sources before dictionary reduction.'
            raise ValueError(msg)
        if self.freq_data is None:
            msg = 'freq_data must be supplied for dictionary reduction.'
            raise ValueError(msg)
        microphones = self.steer.mics.num_mics
        if self.freq_data.num_channels != microphones:
            msg = 'The CSM channel count must match the number of steering-vector microphones.'
            raise ValueError(msg)
        if self.n >= microphones:
            msg = 'n must be smaller than the number of microphones to leave a nonempty noise subspace.'
            raise ValueError(msg)

        frequencies = self.freq_data.fftfreq()
        index = np.searchsorted(frequencies, self.freq)
        if index >= len(frequencies) or frequencies[index] != self.freq:
            msg = f'freq ({self.freq:g} Hz) must be a resolved FFT frequency from freq_data.fftfreq().'
            raise ValueError(msg)

        # Acoular stores eigenvectors in ascending eigenvalue order.
        eigenvectors = np.asarray(self.freq_data.eigvecs[index])
        atoms = self.steer.transfer(frequencies[index]).T
        energies = np.abs(eigenvectors.conj().T @ atoms) ** 2
        signal, noise = self._projection_sums(energies[-self.n :], energies[: -self.n])
        return signal > noise

    def _projection_sums(self, signal, noise):
        """Sum all squared projection coefficients in each subspace."""
        return signal.sum(axis=0), noise.sum(axis=0)


class DRPE(DRAE):
    """Reduce a dictionary using atom-dependent partial eigenvectors (DR-PE).

    Implements Feng et al. (2026), Eqs. (20)-(21),
    https://doi.org/10.1121/10.0046883. Inputs, frequency handling, and assumptions
    are those of :class:`DRAE`, including the explicitly supplied source count
    :attr:`n` and use of the full CSM.

    For each dictionary atom independently, retain the ``min(n, M-n)`` largest
    squared projection coefficients from each eigenvector group, where ``M`` is
    the number of microphones. Retain the atom only if the signal-group sum is
    strictly greater than the noise-group sum. Thus the larger subspace is reduced
    to the dimension of the smaller one; when ``n == M-n``, DR-PE equals DR-AE.
    Selection is based on projection magnitude, not eigenvalue magnitude.

    Notes
    -----
    Partial-eigenvector selection depends on the eigensolver's basis within
    repeated-eigenvalue eigenspaces. In particular, rank-deficient CSMs can have
    non-unique noise eigenvectors. Unlike a projection onto an entire subspace,
    truncation of such a basis is not invariant to rotations within that subspace.
    No special basis is imposed beyond that returned by :attr:`freq_data`.
    """

    def _projection_sums(self, signal, noise):
        """Compare equal counts of the largest coefficients separately for each atom."""
        dimension = min(signal.shape[0], noise.shape[0])
        if signal.shape[0] > dimension:
            signal = np.partition(signal, -dimension, axis=0)[-dimension:]
        if noise.shape[0] > dimension:
            noise = np.partition(noise, -dimension, axis=0)[-dimension:]
        return signal.sum(axis=0), noise.sum(axis=0)
