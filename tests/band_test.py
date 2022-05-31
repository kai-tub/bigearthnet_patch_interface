import numpy as np
import pytest
from pydantic import ValidationError

from bigearthnet_patch_interface.band_interface import *


def test_band_shape_validator():
    with pytest.raises(ValueError):
        Band(name="B01", spatial_resolution=10, data=np.array([1]), data_shape=((2, 1)))


@pytest.mark.parametrize(
    "invalid_data",
    [
        [[1]],
        ((1)),
        1,
        "1",
    ],
)
def test_data_validation(invalid_data):
    with pytest.raises(ValidationError):
        Band(name="B01", spatial_resolution=10, data=invalid_data, data_shape=(1, 1))


def test_str_representation():
    name = "B01"
    sp = 10
    b = Band(name=name, spatial_resolution=sp, data=np.array([[1]]), data_shape=(1, 1))
    assert name in str(b) and str(sp) in str(b)


@pytest.mark.parametrize(
    "invalid_name",
    [
        "wrong_name",
        "B09",
        "B01",
        "B8A",
    ],
)
def test_10m_name_validation(invalid_name):
    with pytest.raises(ValueError):
        BenS2_10mBand(name=invalid_name)


@pytest.mark.parametrize(
    "invalid_name",
    [
        "wrong_name",
        "B02",
        "B03",
        "B09",
    ],
)
def test_20m_name_validation(invalid_name):
    with pytest.raises(ValueError):
        BenS2_20mBand(name=invalid_name)


@pytest.mark.parametrize(
    "invalid_name",
    [
        "wrong_name",
        "B01",
        "B02",
        "B11",
    ],
)
def test_60m_name_validation(invalid_name):
    with pytest.raises(ValueError):
        BenS2_60mBand(name=invalid_name)


@pytest.mark.parametrize(
    "invalid_name",
    [
        "wrong_name",
        "B01",
        "B02",
        "B06",
    ],
)
def test_s1_name_validation(invalid_name):
    with pytest.raises(ValueError):
        BenS1_Band(name=invalid_name)
