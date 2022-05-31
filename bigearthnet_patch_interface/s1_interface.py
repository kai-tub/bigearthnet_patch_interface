import pickle
from typing import Tuple

import natsort
import numpy as np
from pydantic import Field, validate_arguments

from bigearthnet_patch_interface.patch_interface import BigEarthNetPatch

from .band_interface import *
from .patch_interface import *

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


class BigEarthNet_S1_Patch(BigEarthNetPatch):
    def __init__(
        self,
        bandVH: np.ndarray,
        bandVV: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Original BigEarthNet-S1 patch class.
        Will store any additional keyword arguments.

        """
        self.bandVH = BenS1_Band(name="VH", data=bandVH)
        self.bandVV = BenS1_Band(name="VV", data=bandVV)

        bands = [
            self.bandVH,
            self.bandVV,
        ]

        super().__init__(bands=bands, **kwargs)

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
        short_band_name: str = Field(..., min_length=1, max_length=4)
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

    def get_bands(self) -> Tuple[np.ndarray, ...]:
        """
        Get  the _data_ of the S1 bands of the patch interface as a tuple.
        The ordering is guaranteed to be naturally sorted:
        VH, VV

        Returns:
            Tuple[np.ndarray]: Bands: VH, VV
        """
        return tuple(b.data for b in self.bands)

    def get_stacked_bands(self) -> np.ndarray:
        """
        Quick way to get the bands already stacked.
        Calls `np.stack(self.get_bands())`.

        Returns:
            np.ndarray: Stacked VH, VV bands
        """
        return np.stack(self.get_bands())
