
import acoular
import numpy as np
import pytest
from acoular import (
    BeamformerBase,
    BeamformerEig,
    SteeringVector,
)

from tests.setup import SetupSourceCase, test_config

acoular.config.global_caching = 'none'  # to make sure that nothing is cached


# if this flag is set to True
WRITE_NEW_REFERENCE_DATA = False
# new beamformer results are generated for comparison during testing.
# Should always be False. Only set to True if it is necessary to
# recalculate the data due to intended changes of the Beamformers.


STEER_TYPES = {
    'classic' : 1,
    'inverse' : 2,
    'true level' : 3,
    'true location' : 4
}

sc = SetupSourceCase()

bf = [
    BeamformerBase(freq_data=sc.freq_data, steer=sc.steer, cached=False),
    BeamformerEig(freq_data=sc.freq_data, steer=sc.steer, n=54, cached=False)
]

@pytest.mark.parametrize("steer_type, r_diag, bf", [
    (s,r,b) for s in STEER_TYPES for r in (True, False) for b in bf])
def test_all_steer_formulation(steer_type, r_diag, b):
    """tests all variants of beamformerFreq subroutines"""
    st = SteeringVector(grid=sc.grid, mics=sc.mics, env=sc.env, steer_type=steer_type)
    b.r_diag = r_diag
    b.steer = st
    name = test_config.reference_data / f'{b.__class__.__name__}{r_diag}{STEER_TYPES[steer_type]}.npy'
    actual_data = np.array([b.synthetic(cf, 1) for cf in sc.cfreqs], dtype=np.float32)
    if WRITE_NEW_REFERENCE_DATA:
        np.save(name, actual_data)
    ref_data = np.load(name)
    np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

