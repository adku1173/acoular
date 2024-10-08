# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements testing of frequency beamformers."""

import acoular
import numpy as np
import pytest
from acoular import (
    BeamformerBase,
    BeamformerCapon,
    BeamformerClean,
    BeamformerCleansc,
    BeamformerCMF,
    BeamformerDamas,
    BeamformerDamasPlus,
    BeamformerEig,
    BeamformerFunctional,
    BeamformerGIB,
    BeamformerGridlessOrth,
    BeamformerMusic,
    BeamformerOrth,
    BeamformerSODIX,
)

from tests.setup import SetupSourceCase, test_config

acoular.config.global_caching = 'none'  # to make sure that nothing is cached


# if this flag is set to True
WRITE_NEW_REFERENCE_DATA = False
# new beamformer results are generated for comparison during testing.
# Should always be False. Only set to True if it is necessary to
# recalculate the data due to intended changes of the Beamformers.

sc = SetupSourceCase()

# copy CMF classes as workaround so that reference data name is unique later
class BeamformerCMFLassoLarsBIC(BeamformerCMF):
    pass


class BeamformerCMFNNLS(BeamformerCMF):
    pass


# produces a tuple of beamformer objects to test
# because we need new objects for each test we have to call this more than once
def fbeamformers():
    # frequency beamformers to test
    bbase = BeamformerBase(freq_data=sc.freq_data, steer=sc.steer, r_diag=True, cached=False)
    bc = BeamformerCapon(freq_data=sc.freq_data, steer=sc.steer, cached=False)
    beig = BeamformerEig(freq_data=sc.freq_data, steer=sc.steer, r_diag=True, n=54, cached=False)
    bm = BeamformerMusic(freq_data=sc.freq_data, steer=sc.steer, n=6, cached=False)
    bd = BeamformerDamas(freq_data=sc.freq_data, steer=sc.steer, r_diag=True, n_iter=10, cached=False)
    bdp = BeamformerDamasPlus(freq_data=sc.freq_data, steer=sc.steer, r_diag=True, n_iter=100, cached=False)
    bo = BeamformerOrth(freq_data=sc.freq_data, steer=sc.steer, r_diag=True, eva_list=list(range(38, 54)), cached=False)
    bs = BeamformerCleansc(freq_data=sc.freq_data, steer=sc.steer, r_diag=True, cached=False)
    bcmflassobic = BeamformerCMFLassoLarsBIC(freq_data=sc.freq_data, steer=sc.steer, method='LassoLarsBIC', cached=False)
    bcmfnnls = BeamformerCMFNNLS(freq_data=sc.freq_data, steer=sc.steer, method='NNLS', cached=False)
    bl = BeamformerClean(freq_data=sc.freq_data, steer=sc.steer, r_diag=True, n_iter=10, cached=False)
    bf = BeamformerFunctional(freq_data=sc.freq_data, steer=sc.steer, r_diag=False, gamma=3, cached=False)
    bgib = BeamformerGIB(freq_data=sc.freq_data, steer=sc.steer, method='LassoLars', n=2, cached=False)
    bgo = BeamformerGridlessOrth(freq_data=sc.freq_data, steer=sc.steer, r_diag=False, n=1, shgo={'n': 16}, cached=False)
    bsodix = BeamformerSODIX(freq_data=sc.freq_data, steer=sc.steer, max_iter=10, cached=False)
    return (bbase, bc, beig, bm, bl, bo, bs, bd, bcmflassobic, bcmfnnls, bf, bdp, bgib, bgo, bsodix)


@pytest.mark.parametrize("beamformer, cache_option", fbeamformers())
def test_results(b, cache_option):
    acoular.config.global_caching = cache_option
    if cache_option == ['individual', 'all']:
        b.cached = True
    name = test_config.reference_data / f'{b.__class__.__name__}.npy'
    # stack all frequency band results together
    actual_data = np.array([b.synthetic(cf, 1) for cf in sc.cfreqs], dtype=np.float32)
    if WRITE_NEW_REFERENCE_DATA and cache_option == 'none':
        np.save(name, actual_data)
    ref_data = np.load(name)
    np.testing.assert_allclose(actual_data, ref_data, rtol=5e-5, atol=5e-8)
