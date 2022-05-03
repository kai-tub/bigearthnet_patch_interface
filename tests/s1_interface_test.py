import numpy as np
import pytest

from bigearthnet_patch_interface.band_interface import Band
from bigearthnet_patch_interface.s1_interface import *

TEST_BANDS = {
    "bandVV": random_ben_S1_band(),
    "bandVH": random_ben_S1_band(),
}


def test_random_ben_band():
    rand_s1_band = random_ben_S1_band()
    assert rand_s1_band.shape == (120, 120)


def test_s1_patches_nat_order():
    ben_patch = BigEarthNet_S1_Patch(**TEST_BANDS)
    assert (ben_patch.get_bands()[0] == TEST_BANDS["bandVH"]).all()
    assert (ben_patch.get_bands()[1] == TEST_BANDS["bandVV"]).all()


def test_s1_patches_short_init():
    d = {"VV": TEST_BANDS["bandVV"], "VH": TEST_BANDS["bandVH"]}
    ben_patch = BigEarthNet_S1_Patch.short_init(**d)
    assert (ben_patch.get_bands()[0] == TEST_BANDS["bandVH"]).all()
    assert (ben_patch.get_bands()[1] == TEST_BANDS["bandVV"]).all()


def test_wrong_resolution():
    d = TEST_BANDS.copy()
    d["bandVV"] = d["bandVV"].flatten()
    with pytest.raises(ValueError):
        BigEarthNet_S1_Patch(**d)


@pytest.mark.parametrize("name", ["VV", "VH"])
def test_band_by_name(name):
    ben_patch = BigEarthNet_S1_Patch(**TEST_BANDS)
    isinstance(ben_patch.get_band_by_name(name), Band)


@pytest.mark.parametrize("name", ["VV", "VH"])
def test_band_by_name_data(name):
    ben_patch = BigEarthNet_S1_Patch(**TEST_BANDS)
    isinstance(ben_patch.get_band_data_by_name(name), np.ndarray)


@pytest.mark.parametrize("invalid_name", ["B01", "bandVH", "Vh"])
def test_invalid_band_by_name(invalid_name):
    ben_patch = BigEarthNet_S1_Patch(**TEST_BANDS)
    with pytest.raises(KeyError):
        ben_patch.get_band_data_by_name(invalid_name)


def test_stacked_bands():
    ben_patch = BigEarthNet_S1_Patch(**TEST_BANDS)
    stacked_bands = ben_patch.get_stacked_bands()
    assert stacked_bands.shape == (2, 120, 120)
    assert (stacked_bands[0] == TEST_BANDS["bandVH"]).all()


def test_metadata():
    metadata = {"labels": ["Marine waters"]}
    ben_patch = BigEarthNet_S1_Patch(**TEST_BANDS, **metadata)
    assert ben_patch.labels == metadata["labels"]


def test_pickle():
    ben_patch = BigEarthNet_S1_Patch(**TEST_BANDS)
    p_data = ben_patch.dumps()
    ben_patch_2 = BigEarthNet_S1_Patch.loads(p_data)
    assert (ben_patch.get_stacked_bands() == ben_patch_2.get_stacked_bands()).all()


@pytest.mark.parametrize(
    ("inp_name", "out_name"),
    [
        ("VV", "bandVV"),
        ("VH", "bandVH"),
        ("Vh", "bandVH"),
        ("vv", "bandVV"),
    ],
)
def test_short_to_long_name(inp_name, out_name):
    assert BigEarthNet_S1_Patch.short_to_long_band_name(inp_name) == out_name
