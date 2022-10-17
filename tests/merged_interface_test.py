import numpy as np
import pytest

from bigearthnet_patch_interface.band_interface import Band
from bigearthnet_patch_interface.merged_interface import *
from bigearthnet_patch_interface.s1_interface import *
from bigearthnet_patch_interface.s2_interface import *

TEST_BANDS = {
    "bandVV": random_ben_S1_band(),
    "bandVH": random_ben_S1_band(),
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


def test_short_init_merged_patches():
    d = {
        "B01": TEST_BANDS["band01"],
        "B02": TEST_BANDS["band02"],
        "B03": TEST_BANDS["band03"],
        "B04": TEST_BANDS["band04"],
        "B05": TEST_BANDS["band05"],
        "B06": TEST_BANDS["band06"],
        "B07": TEST_BANDS["band07"],
        "B08": TEST_BANDS["band08"],
        "B8A": TEST_BANDS["band8A"],
        "B09": TEST_BANDS["band09"],
        "B11": TEST_BANDS["band11"],
        "B12": TEST_BANDS["band12"],
        "VV": TEST_BANDS["bandVV"],
        "VH": TEST_BANDS["bandVH"],
    }
    ben_patch = BigEarthNet_S1_S2_Patch.short_init(**d)
    assert (ben_patch.get_band_data_by_name("B02") == TEST_BANDS["band02"]).all()
    assert (ben_patch.get_band_data_by_name("B05") == TEST_BANDS["band05"]).all()
    assert (ben_patch.get_band_data_by_name("VV") == TEST_BANDS["bandVV"]).all()
    assert (ben_patch.get_band_data_by_name("VH") == TEST_BANDS["bandVH"]).all()


@pytest.mark.parametrize("name", ["B01", "B04", "B11", "VV"])
def test_band_by_name(name):
    ben_patch = BigEarthNet_S1_S2_Patch(**TEST_BANDS)
    isinstance(ben_patch.get_band_by_name(name), Band)


def test_metadata():
    metadata = {"labels": ["Marine waters"]}
    ben_patch = BigEarthNet_S1_S2_Patch(**TEST_BANDS, **metadata)
    assert ben_patch.labels == metadata["labels"]


def test_repr():
    ben_patch = BigEarthNet_S1_S2_Patch(**TEST_BANDS)
    s = str(ben_patch)
    assert "spatial resolution" in s


def test_patches_nat_order():
    ben_patch = BigEarthNet_S1_S2_Patch(**TEST_BANDS)
    assert tuple(a.data for a in ben_patch.bands) == (
        TEST_BANDS["band01"],
        TEST_BANDS["band02"],
        TEST_BANDS["band03"],
        TEST_BANDS["band04"],
        TEST_BANDS["band05"],
        TEST_BANDS["band06"],
        TEST_BANDS["band07"],
        TEST_BANDS["band08"],
        TEST_BANDS["band8A"],
        TEST_BANDS["band09"],
        TEST_BANDS["band11"],
        TEST_BANDS["band12"],
        TEST_BANDS["bandVH"],
        TEST_BANDS["bandVV"],
    )


def test_pickle_unpickle_print():
    ben_patch = BigEarthNet_S1_S2_Patch(**TEST_BANDS)
    prev_repr = ben_patch.__repr__()
    d = ben_patch.dumps()
    del ben_patch
    ben_patch_2 = BigEarthNet_S1_S2_Patch.loads(d)
    assert prev_repr == ben_patch_2.__repr__()
