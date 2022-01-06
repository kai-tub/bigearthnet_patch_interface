import pytest
import numpy as np
import pickle

from bigearthnet_patch_interface.s2_interface import *

TEST_BANDS = {
    "band02": random_ben_S2_band(spatial_resoluion=10),
    "band03": random_ben_S2_band(spatial_resoluion=10),
    "band04": random_ben_S2_band(spatial_resoluion=10),
    "band08": random_ben_S2_band(spatial_resoluion=10),
    "band05": random_ben_S2_band(spatial_resoluion=20),
    "band06": random_ben_S2_band(spatial_resoluion=20),
    "band07": random_ben_S2_band(spatial_resoluion=20),
    "band8A": random_ben_S2_band(spatial_resoluion=20),
    "band11": random_ben_S2_band(spatial_resoluion=20),
    "band12": random_ben_S2_band(spatial_resoluion=20),
    "band01": random_ben_S2_band(spatial_resoluion=60),
    "band09": random_ben_S2_band(spatial_resoluion=60),
}


def test_default_s2_patches():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    assert (ben_patch.get_10m_bands()[0] == TEST_BANDS["band02"]).all()
    assert (ben_patch.get_20m_bands()[0] == TEST_BANDS["band05"]).all()
    assert (ben_patch.get_60m_bands()[0] == TEST_BANDS["band01"]).all()


def test_wrong_resolution():
    d = TEST_BANDS.copy()
    d["band01"], d["band02"] = d["band02"], d["band01"]
    with pytest.raises(ValueError):
        BigEarthNet_S2_Patch(**d)


@pytest.mark.parametrize("name", ["B01", "B04", "B11"])
def test_band_by_name(name):
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    isinstance(ben_patch.get_band_by_name(name), np.ndarray)


@pytest.mark.parametrize("invalid_name", ["B1", "B10", "band01"])
def test_invalid_band_by_name(invalid_name):
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    with pytest.raises(KeyError):
        ben_patch.get_band_by_name(invalid_name)


def test_stacked_10m_bands():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    stacked_10m_bands = ben_patch.get_stacked_10m_bands()
    assert stacked_10m_bands.shape == (4, 120, 120)
    assert (stacked_10m_bands[0] == TEST_BANDS["band02"]).all()


def test_stacked_20m_bands():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    stacked_20m_bands = ben_patch.get_stacked_20m_bands()
    assert stacked_20m_bands.shape == (6, 60, 60)
    assert (stacked_20m_bands[0] == TEST_BANDS["band05"]).all()


def test_stacked_60m_bands():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    stacked_60m_bands = ben_patch.get_stacked_60m_bands()
    assert stacked_60m_bands.shape == (2, 20, 20)
    assert (stacked_60m_bands[0] == TEST_BANDS["band01"]).all()


def test_metadata():
    metadata = {"labels": ["Marine waters"]}
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS, **metadata)
    assert ben_patch.labels == metadata["labels"]


def test_pickle():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    p_data = ben_patch.dumps()
    ben_patch_2 = BigEarthNet_S2_Patch.loads(p_data)
    assert (
        ben_patch.get_stacked_10m_bands() == ben_patch_2.get_stacked_10m_bands()
    ).all()
