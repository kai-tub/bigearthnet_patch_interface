import pickle

import numpy as np

from bigearthnet_patch_interface.patch_interface import BigEarthNetPatch

from .band_interface import *
from .patch_interface import *
from .s1_interface import BigEarthNet_S1_Patch
from .s2_interface import BigEarthNet_S2_Patch


class BigEarthNet_S1_S2_Patch(BigEarthNetPatch):
    def __init__(
        self,
        bandVH: np.ndarray,
        bandVV: np.ndarray,
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
        self.s1_patch = BigEarthNet_S1_Patch(bandVH=bandVH, bandVV=bandVV)
        self.s2_patch = BigEarthNet_S2_Patch(
            band01=band01,
            band02=band02,
            band03=band03,
            band04=band04,
            band05=band05,
            band06=band06,
            band07=band07,
            band08=band08,
            band8A=band8A,
            band09=band09,
            band11=band11,
            band12=band12,
        )

        self._bands = [*self.s1_patch.bands, *self.s2_patch.bands]
        super().store_kwargs_as_props(**kwargs)

    @classmethod
    def short_init(
        cls,
        VH: np.ndarray,
        VV: np.ndarray,
        B01: np.ndarray,
        B02: np.ndarray,
        B03: np.ndarray,
        B04: np.ndarray,
        B05: np.ndarray,
        B06: np.ndarray,
        B07: np.ndarray,
        B08: np.ndarray,
        B8A: np.ndarray,
        B09: np.ndarray,
        B11: np.ndarray,
        B12: np.ndarray,
        **kwargs,
    ):
        """
        Alternative `__init__` function.
        Only difference is the encoded names.
        """
        return cls(
            bandVH=VH,
            bandVV=VV,
            band01=B01,
            band02=B02,
            band03=B03,
            band04=B04,
            band05=B05,
            band06=B06,
            band07=B07,
            band08=B08,
            band8A=B8A,
            band09=B09,
            band11=B11,
            band12=B12,
            **kwargs,
        )
