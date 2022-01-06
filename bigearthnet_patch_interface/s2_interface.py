from os import stat
import pickle
from typing import Tuple

import natsort
import numpy as np
from .band_interface import *

__all__ = [
    "BigEarthNet_S2_Patch",
    "S2_DN_TO_REFLECTANCE",
    "random_ben_S2_band",
    "s2_to_float",
    "s2_to_float32",
    "s2_to_float64",
]

S2_DN_TO_REFLECTANCE = 10_000


def _s2_to_float(arr):
    return arr / S2_DN_TO_REFLECTANCE


def s2_to_float64(arr) -> np.ndarray:
    return np.float64(_s2_to_float(arr))


def s2_to_float32(arr) -> np.ndarray:
    return np.float32(_s2_to_float(arr))


def s2_to_float(arr) -> np.ndarray:
    """
    Convert to numpy float
    """
    return s2_to_float32(arr)


def random_ben_S2_band(spatial_resoluion=10, original_dtype=False):
    BEN_PATCH_SIZE = 1200
    pixel_resolution = BEN_PATCH_SIZE // spatial_resoluion
    arr = np.random.randint(
        S2_DN_TO_REFLECTANCE, size=(pixel_resolution, pixel_resolution), dtype="uint16"
    )
    if not original_dtype:
        return s2_to_float(arr)
    return arr


class BigEarthNet_S2_Patch:
    def __init__(
        self,
        band01: np.ndarray,
        band02: np.ndarray,
        band03: np.ndarray,
        band04: np.ndarray,
        band05: np.ndarray,
        band06: np.ndarray,
        band07: np.ndarray,
        band08: np.ndarray,
        band8A: np.ndarray,
        band09: np.ndarray,
        band11: np.ndarray,
        band12: np.ndarray,
        **kwargs,
    ):
        """
        Original BigEarthNet-S2 patch class.
        Will store any additional keyword arguments.

        """
        self.band01 = BenS2_60mBand(name="B01", data=band01)
        self.band02 = BenS2_10mBand(name="B02", data=band02)
        self.band03 = BenS2_10mBand(name="B03", data=band03)
        self.band04 = BenS2_10mBand(name="B04", data=band04)
        self.band05 = BenS2_20mBand(name="B05", data=band05)
        self.band06 = BenS2_20mBand(name="B06", data=band06)
        self.band07 = BenS2_20mBand(name="B07", data=band07)
        self.band08 = BenS2_10mBand(name="B08", data=band08)
        self.band8A = BenS2_20mBand(name="B8A", data=band8A)
        self.band09 = BenS2_60mBand(name="B09", data=band09)
        self.band11 = BenS2_20mBand(name="B11", data=band11)
        self.band12 = BenS2_20mBand(name="B12", data=band12)

        self.bands = [
            self.band01,
            self.band02,
            self.band03,
            self.band04,
            self.band05,
            self.band06,
            self.band07,
            self.band08,
            self.band8A,
            self.band09,
            self.band11,
            self.band12,
        ]

        # guarantee that the bands are naturally sorted
        # which is important for quick filtering operations!
        self._bands = natsort.natsorted(self.bands, key=lambda band: band.name)
        # store extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__stored_args__ = {**kwargs}

    def dump(self, file):
        return pickle.dump(self, file, protocol=4)

    def dumps(self):
        return pickle.dumps(self, protocol=4)

    @staticmethod
    def load(file) -> "BigEarthNet_S2_Patch":
        return pickle.load(file)

    @staticmethod
    def loads(data) -> "BigEarthNet_S2_Patch":
        return pickle.loads(data)

    def get_band_by_name(self, name: str) -> np.ndarray:
        band = None
        for b in self._bands:
            if b.name == name:
                band = b
        if band is None:
            raise KeyError(f"{name} is not known")
        return band

    def get_10m_bands(self) -> Tuple[np.ndarray]:
        return tuple(b.data for b in self.bands if b.spatial_resolution == 10)

    def get_20m_bands(self) -> Tuple[np.ndarray]:
        return tuple(b.data for b in self.bands if b.spatial_resolution == 20)

    def get_60m_bands(self) -> Tuple[np.ndarray]:
        return tuple(b.data for b in self.bands if b.spatial_resolution == 60)

    def get_stacked_10m_bands(self) -> np.ndarray:
        return np.stack(self.get_10m_bands())

    def get_stacked_20m_bands(self) -> np.ndarray:
        return np.stack(self.get_20m_bands())

    def get_stacked_60m_bands(self) -> np.ndarray:
        return np.stack(self.get_60m_bands())

    def __repr__(self):
        r_str = f"{self.__class__.__name__} with:\n"
        r_str += "\n".join(f"\t{b}" for b in self._bands)
        if len(self.__stored_args__) != 0:
            r_str += "\nAnd the extra metadata:\n"
            for key, metadata in self.__stored_args__.items():
                r_str += f"\t{key}: {metadata}\n"
        return r_str
