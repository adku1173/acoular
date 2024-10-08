import acoular as ac
import numpy as np
import pytest
from tests.setup import SetupSourceCase

sc = SetupSourceCase()
tc = ac.TimeCache(source=sc.source)

@pytest.mark.parametrize("conf", ['individual', 'all', 'none', 'readonly', 'overwrite'])
def test_valid_cache_result(conf):
    """manually create an incomplete cash file and then read it."""
    block_size = 5
    ac.config.global_caching = conf
    for i, (block_c, block_nc) in enumerate(zip(tc.result(block_size), sc.source.result(block_size))):
        np.testing.assert_array_almost_equal(block_c, block_nc)
        if i == 0:
            break
    for block_c, block_nc in zip(tc.result(block_size), sc.source.result(block_size)):
        np.testing.assert_array_almost_equal(block_c, block_nc)

