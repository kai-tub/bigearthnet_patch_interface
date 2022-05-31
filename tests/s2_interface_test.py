import numpy as np
import pytest

from bigearthnet_patch_interface.band_interface import Band
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


def test_S2_to_float32():
    a = np.array([1, 2, 3])
    r = s2_to_float32(a)
    assert r.dtype == "float32"


def test_S2_to_float64():
    a = np.array([1, 2, 3])
    r = s2_to_float64(a)
    assert r.dtype == "float64"


def test_S2_to_float():
    a = np.array([1, 2, 3])
    r = s2_to_float(a)
    assert r.dtype == "float32"


@pytest.mark.parametrize(
    ("spatial_res", "output_shape"),
    [
        (10, (120, 120)),
        (20, (60, 60)),
        (60, (20, 20)),
    ],
)
def test_random_ben_band(spatial_res, output_shape):
    assert random_ben_S2_band(spatial_resoluion=spatial_res).shape == output_shape


def test_default_s2_patches():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    assert (ben_patch.get_10m_bands()[0] == TEST_BANDS["band02"]).all()
    assert (ben_patch.get_20m_bands()[0] == TEST_BANDS["band05"]).all()
    assert (ben_patch.get_60m_bands()[0] == TEST_BANDS["band01"]).all()


def test_short_init_s2_patches():
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
    }
    ben_patch = BigEarthNet_S2_Patch.short_init(**d)
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
    isinstance(ben_patch.get_band_data_by_name(name), Band)


@pytest.mark.parametrize("name", ["B01", "B04", "B11"])
def test_band_by_name_data(name):
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    isinstance(ben_patch.get_band_data_by_name(name), np.ndarray)


@pytest.mark.parametrize("invalid_name", ["B1", "B10", "band01"])
def test_invalid_band_by_name(invalid_name):
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    with pytest.raises(KeyError):
        ben_patch.get_band_data_by_name(invalid_name)


def test_10_bands_order():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    tup_10m_bands = ben_patch.get_10m_bands()
    assert tuple(arr for arr in tup_10m_bands) == (
        TEST_BANDS["band02"],
        TEST_BANDS["band03"],
        TEST_BANDS["band04"],
        TEST_BANDS["band08"],
    )


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


def test_20_bands_order():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    tup_20m_bands = ben_patch.get_20m_bands()
    assert tuple(arr for arr in tup_20m_bands) == (
        TEST_BANDS["band05"],
        TEST_BANDS["band06"],
        TEST_BANDS["band07"],
        TEST_BANDS["band8A"],
        TEST_BANDS["band11"],
        TEST_BANDS["band12"],
    )


def test_60_bands_order():
    ben_patch = BigEarthNet_S2_Patch(**TEST_BANDS)
    tup_60m_bands = ben_patch.get_60m_bands()
    assert tuple(arr for arr in tup_60m_bands) == (
        TEST_BANDS["band01"],
        TEST_BANDS["band09"],
    )


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


@pytest.mark.parametrize(
    ("inp_name", "out_name"),
    [
        ("B01", "band01"),
        ("b01", "band01"),
        ("B1", "band01"),
        ("01", "band01"),
        ("1", "band01"),
        ("B08A", "band8A"),
        ("B8A", "band8A"),
        ("08A", "band8A"),
        ("8A", "band8A"),
    ],
)
def test_short_to_long_name(inp_name, out_name):
    assert BigEarthNet_S2_Patch.short_to_long_band_name(inp_name) == out_name


# def test_interp():
#     import torch.nn

#     r = torch.rand(1, 120, 120)

#     # bands = torch.unsqueeze(r, 1)  # input for bicubic must be 4 dimensional, e.g. from (6,60,60) to (6,1,60,60)
#     res1 = torch.nn.functional.interpolate(
#         torch.unsqueeze(r, 1), [120, 120], mode="bicubic"
#     )
#     res2 = torch.nn.functional.interpolate(
#         torch.unsqueeze(r, 0), [120, 120], mode="bicubic"
#     )
#     print(res1)
#     print(res2)
#     # assert (res1.flatten() == res2.flatten()).all()
#     assert False
