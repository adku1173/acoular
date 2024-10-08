import tempfile
from copy import deepcopy
from pathlib import Path

import acoular as ac
import h5py
import numpy as np
import pytest
import tables

from tests.setup import test_config

#TODO: readonly may not make sense. Should be handled in the test setup (is [:] copy handled correctly?)

CACHE_TEST_PARAMS = [
    ("none", True, 'h5py'), ("none", False, 'h5py'), ("none", True, 'tables'), ("none", False, 'tables'),
    ("individual", True, 'h5py'), ("individual", False, 'h5py'), ("individual", True, 'tables'), ("individual", False, 'tables'),
    ("all", True, 'h5py'), ("all", False, 'h5py'), ("all", True, 'tables'), ("all", False, 'tables'),
    ("overwrite", True, 'h5py'), ("overwrite", False, 'h5py'), ("overwrite", True, 'tables'), ("overwrite", False, 'tables'),
    ("readonly", True, 'h5py'), ("readonly", False, 'h5py'), ("readonly", True, 'tables'), ("readonly", False, 'tables'),
    ]


array_lib = {
    "h5py": h5py.Dataset,
    "tables": tables.carray.CArray,
}

@pytest.fixture
def power_spectra(source_case, global_caching, cached, lib):
    ac.config.h5library = lib
    ac.config.global_caching = global_caching
    name = test_config.reference_data / 'PowerSpectra_csm.npy'
    ref_data = np.load(name)
    power_spectra = source_case.freq_data
    power_spectra.cached = cached
    power_spectra.default_cache_name = Path(tempfile.mktemp()).name + '.h5' # create unique cache file name for each test
    if global_caching == "none":
        yield power_spectra, ref_data, np.ndarray
    if global_caching == "individual":
        array_class = np.ndarray if not cached else array_lib[lib]
        yield power_spectra, ref_data, array_class
    if global_caching == "all":
        array_class = array_lib[lib]
        yield power_spectra, ref_data, array_class
    if global_caching == "overwrite":
        array_class = array_lib[lib]
        yield power_spectra, ref_data, array_class
    if global_caching == "readonly":
        array_class = np.ndarray # if the cache does not exist!
        yield power_spectra, ref_data, array_class
    del source_case # delete the object after the test


@pytest.fixture
def beamformer(source_case, bf, global_caching, cached, lib):
    beamformer = bf()
    ac.config.h5library = lib
    ac.config.global_caching = global_caching
    name = test_config.reference_data / f'{beamformer.__class__.__name__}.npy'
    ref_data = np.load(name)
    freq_data = source_case.freq_data
    freq_data.cached = False
    beamformer.default_cache_name = Path(tempfile.mktemp()).name + '.h5' # create unique cache file name for each test
    beamformer.freq_data = freq_data
    beamformer.steer = source_case.steer
    if global_caching == "none":
        yield beamformer, ref_data, np.ndarray
    if global_caching == "individual":
        array_class = np.ndarray if not cached else array_lib[lib]
        yield beamformer, ref_data, array_class
    if global_caching == "all":
        array_class = array_lib[lib]
        yield beamformer, ref_data, array_class
    if global_caching == "overwrite":
        array_class = array_lib[lib]
        yield beamformer, ref_data, array_class
    if global_caching == "readonly":
        array_class = np.ndarray # if the cache does not exist!
        yield beamformer, ref_data, array_class
    del source_case # delete the object after the test


@pytest.mark.parametrize("global_caching, cached, lib", CACHE_TEST_PARAMS)
def test_cache_power_spectra(power_spectra):
    ps, ref_data, array_class = power_spectra
    assert ps.h5f is None # we make sure that the file is not opened already
    csm = ps.csm # calculate the cross spectral matrix
    addr = id(csm)
    actual_data = np.array(csm[(16, 32), :, :], dtype=np.complex64)
    assert isinstance(csm, array_class) # check if the array is referenced in the cache or is a numpy array
    np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

    if (ac.config.global_caching != "none") and (ac.config.h5library == "tables"):
        if ac.config.global_caching == "overwrite":
            ps._csm[:] = 0 # manipulate the csm
        ps_copy = deepcopy(ps)
        # remove traits cache (not the cache file!!)
        if ps_copy.__dict__.get("_traits_cache_csm"):
            del ps_copy.__dict__["_traits_cache_csm"]
        if ps_copy.h5f:
            ps_copy.h5f.close()
            ps_copy.h5f = None
        csm_from_cache_file = ps_copy.csm # read the csm from the cache file (if it exists)

        # assert that the csm is not the same object as the one calculated before when overwriting
        if ac.config.global_caching == "overwrite":
            assert id(csm_from_cache_file) != addr # has changed after the overwrite

        if ac.config.global_caching == "individual":
            if ps_copy.cached:
                assert csm_from_cache_file is csm
            else:
                assert csm_from_cache_file is not csm

        if ac.config.global_caching == "all":
            assert csm_from_cache_file is csm

        actual_data = np.array(csm_from_cache_file[(16, 32), :, :], dtype=np.complex64)
        np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)


def BeamformerBase_():
    return ac.BeamformerBase(r_diag=True)

BF_PARAMS = [(BeamformerBase_,) + cp for cp in CACHE_TEST_PARAMS]
@pytest.mark.parametrize("bf, global_caching, cached, lib", BF_PARAMS)
def test_cache_beamformer(beamformer):
    bf, ref_data, array_class = beamformer
    assert bf.h5f is None # we make sure that the file is not opened already
    result = bf.result # calculate the cross spectral matrix
    addr = id(result)
    actual_data = actual_data = np.array([bf.synthetic(cf, 1) for cf in (1000, 8000)], dtype=np.float32)
    assert isinstance(bf._ac, array_class) # check if the array is referenced in the cache or is a numpy array
    np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

    if (ac.config.global_caching != "none") and (ac.config.h5library == "tables"):
        if ac.config.global_caching == "overwrite":
            bf.get_cache()[0][:] = 0 # manipulate the result
        bf_copy = deepcopy(bf)
        # remove traits cache (not the cache file!!)
        if bf_copy.__dict__.get("_traits_cache_result"):
            del bf_copy.__dict__["_traits_cache_result"]
        if bf_copy.h5f:
            bf_copy.h5f.close()
            bf_copy.h5f = None
        result_from_cache_file = bf_copy.result # read the result from the cache file (if it exists)

        # assert that the result is not the same object as the one calculated before when overwriting
        if ac.config.global_caching == "overwrite":
            assert id(result_from_cache_file) != addr # has changed after the overwrite

        if ac.config.global_caching == "individual":
            if bf_copy.cached:
                assert result_from_cache_file is result
            else:
                assert result_from_cache_file is not result

        if ac.config.global_caching == "all":
            assert result_from_cache_file is result

        actual_data = np.array([bf_copy.synthetic(cf, 1) for cf in (1000, 8000)], dtype=np.float32)
        np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)



