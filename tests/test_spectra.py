import acoular
import numpy as np

from tests.setup import SetupSourceCase, test_config

acoular.config.global_caching = 'none'  # to make sure that nothing is cached

# if this flag is set to True
WRITE_NEW_REFERENCE_DATA = False
# new beamformer results are generated for comparison during testing.
# Should always be False. Only set to True if it is necessary to
# recalculate the data due to intended changes of the Beamformers.

sc = SetupSourceCase()
rng = np.random.RandomState(2)
mg = acoular.MicGeom(mpos_tot=rng.normal(0, 1, 3 * 2).reshape(3, 2))
sig = acoular.WNoiseGenerator(seed=1, numsamples=1010, sample_freq=1000)
p = acoular.PointSource(signal=sig, loc=(0, 0, 0), mics=mg)
fft = acoular.FFTSpectra(source=p, window='Hanning', block_size=128)

def test_calc_fft(): #TODO: same style as in other tests (use power spectra output for comparison?)
    """test that fft result has not changed over different releases."""
    fft_sum = 0.3690552444079589 + 0.05264645224899272j
    test_fft_sum = 0
    for temp in fft.result():
        test_fft_sum += temp.sum()
    np.testing.assert_almost_equal(test_fft_sum, fft_sum)


def test_power_spectra_csm():
    """test that csm result has not changed over different releases"""
    name = test_config.reference_data / f'{sc.freq_data.__class__.__name__}_csm.npy'
    # test only two frequencies
    actual_data = np.array(sc.freq_data.csm[(16, 32), :, :], dtype=np.complex64)
    if WRITE_NEW_REFERENCE_DATA:
        np.save(name, actual_data)
    ref_data = np.load(name)
    np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

def test_power_spectra_ev():
    """test that eve and eva result has not changed over different releases"""
    f = sc.freq_data
    name = test_config.reference_data / f'{f.__class__.__name__}_ev.npy'
    # test only two frequencies
    actual_data = np.array((f.eve * f.eva[:, :, np.newaxis])[(16, 32), :, :], dtype=np.complex64)
    if WRITE_NEW_REFERENCE_DATA:
        np.save(name, actual_data)
    ref_data = np.load(name)
    np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

