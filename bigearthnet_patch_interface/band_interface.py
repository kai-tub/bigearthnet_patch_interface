from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, validator

__all__ = ["Band", "BenS2_10mBand", "BenS2_20mBand", "BenS2_60mBand", "BenS1_Band"]


class Band(BaseModel):
    name: str
    spatial_resolution: int
    data_shape: Tuple[int, int]
    data: np.ndarray

    @validator("data")
    def _validate_dimension(cls, v, values):
        data_shape = values["data_shape"]

        if v.shape != data_shape:
            raise ValueError(
                f"Input data has the wrong shape: {v.shape} instead of {data_shape}"
            )
        return v

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f"{self.name} with {self.spatial_resolution}m spatial resolution and a size of {self.data_shape} pixel"


class BenS1_Band(Band):
    name: str
    spatial_resolution: int = 10
    data_shape: Tuple[int, int] = (120, 120)

    @validator("name")
    def _validate_name(cls, v):
        ben_s1_bands = ("VV", "VH")
        if v not in ben_s1_bands:
            raise ValueError(f"Band name must one of {ben_s1_bands}\nGiven: {v}")
        return v


class BenS2_10mBand(Band):
    name: str
    spatial_resolution: int = 10
    data_shape: Tuple[int, int] = (120, 120)

    @validator("name")
    def _validate_name(cls, v):
        ben10m_bands = ("B02", "B03", "B04", "B08")
        if v not in ben10m_bands:
            raise ValueError(f"Band name must one of {ben10m_bands}\nGiven: {v}")
        return v


class BenS2_20mBand(Band):
    spatial_resolution: int = 20
    data_shape: Tuple[int, int] = (60, 60)

    @validator("name")
    def _validate_name(cls, v):
        ben20m_bands = ("B05", "B06", "B07", "B8A", "B11", "B12")
        if v not in ben20m_bands:
            raise ValueError(f"Band name must one of {ben20m_bands}\nGiven: {v}")
        return v


class BenS2_60mBand(Band):
    spatial_resolution: int = 60
    data_shape: Tuple[int, int] = (20, 20)

    @validator("name")
    def _validate_name(cls, v):
        ben20m_bands = ("B01", "B09")
        if v not in ben20m_bands:
            raise ValueError(f"Band name must one of {ben20m_bands}\nGiven: {v}")
        return v
