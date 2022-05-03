import pickle
from typing import Tuple

import natsort
import numpy as np
from pydantic import constr, validate_arguments

from .band_interface import *
from .band_interface import BenS1_Band

__all__ = [
    "BigEarthNet_S1_Patch",
    "random_ben_S1_band",
]


def random_ben_S1_band():
    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/definitions
    # https://docs.sentinel-hub.com/api/latest/data/sentinel-1-grd/

    # linear is the linear power in the chosen backscatter coefficient
    # ususally in the range [0, 0.5] but may reach values of 1_000!
    # no real value
    # is already in float and converted to decibel!
    # db = 10 * log10(linear)
    BEN_PATCH_SIZE = 1200
    SPATIAL_RESOLUTION = 10  # for both channels VV and VH
    pixel_resolution = BEN_PATCH_SIZE // SPATIAL_RESOLUTION
    # typical range is 0 - 1
    arr = np.random.rand(pixel_resolution, pixel_resolution)
    # resulting values are now lower than 0
    db_arr = 10 * np.log10(arr + 1e-6)
    return db_arr


class BigEarthNet_S1_Patch:
    def __init__(
        self,
        bandVH: np.ndarray,
        bandVV: np.ndarray,
        **kwargs,
    ):
        """
        Original BigEarthNet-S1 patch class.
        Will store any additional keyword arguments.

        """
        self.bandVH = BenS1_Band(name="VH", data=bandVH)
        self.bandVV = BenS1_Band(name="VV", data=bandVV)

        self.bands = [
            self.bandVH,
            self.bandVV,
        ]

        # guarantee that the bands are naturally sorted
        # which is important for quick filtering operations!
        self.bands = natsort.natsorted(self.bands, key=lambda band: band.name)
        # store extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__stored_args__ = {**kwargs}

    @classmethod
    def short_init(cls, VH: np.ndarray, VV: np.ndarray, **kwargs):
        """
        Alternative `__init__` function.
        Only difference is the encoded names.
        """
        return cls(bandVH=VH, bandVV=VV, **kwargs)

    @staticmethod
    @validate_arguments
    def short_to_long_band_name(
        short_band_name: constr(min_length=1, max_length=4)
    ) -> str:
        """
        Convert a short input band name, such as VV, Vv, or vh
        to the long band name required by the `__init__` function.

        Args:
            short_band_name (str): Short band name
        """
        short_upper = short_band_name.upper()
        # if for whatever reason provided with leading B
        band = short_upper.lstrip("B")
        return f"band{band}"

    def dump(self, file):
        return pickle.dump(self, file, protocol=4)

    def dumps(self):
        return pickle.dumps(self, protocol=4)

    @staticmethod
    def load(file) -> "BigEarthNet_S1_Patch":
        return pickle.load(file)

    @staticmethod
    def loads(data) -> "BigEarthNet_S1_Patch":
        return pickle.loads(data)

    def get_band_by_name(self, name: str) -> Band:
        band = None
        for b in self.bands:
            if b.name == name:
                band = b
        if band is None:
            raise KeyError(f"{name} is not known")
        return band

    def get_band_data_by_name(self, name: str) -> np.ndarray:
        band = self.get_band_by_name(name)
        return band.data

    def get_bands(self) -> Tuple[np.ndarray]:
        return tuple(b.data for b in self.bands)

    def get_stacked_bands(self) -> np.ndarray:
        return np.stack(self.get_bands())

    def __repr__(self):
        r_str = f"{self.__class__.__name__} with:\n"
        r_str += "\n".join(f"\t{b}" for b in self.bands)
        if len(self.__stored_args__) != 0:
            r_str += "\nAnd the extra metadata:\n"
            for key, metadata in self.__stored_args__.items():
                r_str += f"\t{key}: {metadata}\n"
        return r_str
