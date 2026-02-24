import warnings

import numpy as np
import pytest

from acoular.tools.utils import synthetic_indices

def test_synthetic_indices_num0_exact():
    freqs = np.array([100.0, 200.0, 300.0])
    indices = synthetic_indices(freqs, 200.0, num=0)
    assert indices == [(1, 2)]


def test_synthetic_indices_num0_out_of_range_warns():
    freqs = np.array([100.0, 200.0, 300.0])
    with pytest.warns(Warning, match=r"not in resolved frequency range"):
        indices = synthetic_indices(freqs, 400.0, num=0)
    assert indices == [(3, 3)]



def test_synthetic_indices_num0_offgrid_warns():
    freqs = np.array([100.0, 200.0, 300.0])
    with pytest.warns(Warning, match=r"not in set of discrete FFT sample frequencies"):
        indices = synthetic_indices(freqs, 250.0, num=0)
    assert indices == [(2, 3)]


def test_synthetic_indices_band_empty_warns():
    freqs = np.array([100.0, 200.0, 300.0])
    with pytest.warns(Warning, match=r"does not include any discrete FFT sample frequencies"):
        indices = synthetic_indices(freqs, 150.0, num=3)
    assert indices == [(1, 1)]


def test_synthetic_indices_band_multiple_centers_no_warns():
    freqs = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        indices = synthetic_indices(freqs, [200.0, 250.0], num=1)
    assert caught == []
    assert indices == [(1, 4), (2, 5)]
