import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar

import natsort
import numpy as np

from .band_interface import *

TPatch = TypeVar("TPatch", bound="BigEarthNetPatch")

__all__ = ["BigEarthNetPatch"]


class BigEarthNetPatch(ABC):
    """
    A general BigEarthNet Patch interface.
    It provides commonly used functions for S1/S2 patches.
    It requires each instance to overwrite the `_bands` class variable,
    as this is used to return the naturally sorted `bands` property.
    """

    _bands = NotImplemented

    @property
    def bands(self) -> List[Band]:
        _bands: List[Band] = self._bands
        if _bands == NotImplemented:
            raise NotImplementedError("You must overwrite the `_bands` class variable!")
        return natsort.natsorted(_bands, key=lambda band: band.name)

    def store_kwargs_as_props(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__stored_args__ = {**kwargs}

    def get_band_by_name(self, name: str) -> Band:
        """
        Access a specific band by looking up the name.

        Args:
            name (str): Name of the band (B01, B8A, B10, etc.)

        Raises:
            KeyError: If name cannot be found.

        Returns:
            Band: Selected band.
        """
        band = None
        for b in self.bands:
            if b.name == name:
                band = b
        if band is None:
            raise KeyError(f"{name} is not known")
        return band

    def get_band_data_by_name(self, name: str) -> np.ndarray:
        """
        Get the numpy array of the selected band by looking up the name.
        Calls `get_band_by_name` under the hood.

        Returns:
            np.ndarray: Band data
        """
        band = self.get_band_by_name(name)
        return band.data

    def get_natsorted_bands_by_spatial_res(
        self, spatial_resolution: int
    ) -> Tuple[np.ndarray]:
        return tuple(
            b.data for b in self.bands if b.spatial_resolution == spatial_resolution
        )

    def dump(self, file) -> None:
        """
        Convert the instance to a `pickle` binary stream and write to `file`.
        Will call `pickle.dump` with `protocol=4` internally.
        """
        return pickle.dump(self, file, protocol=4)

    def dumps(self) -> bytes:
        """
        Convert the instance to a `pickle` binary stream.
        Will call `pickle.dump` with `protocol=4` internally.
        """
        return pickle.dumps(self, protocol=4)

    @staticmethod
    def load(file) -> TPatch:
        """
        Calls `pickle.load` under the hood.
        """
        return pickle.load(file)

    @staticmethod
    def loads(data) -> TPatch:
        """
        Calls `pickle.loads` under the hood.
        """
        return pickle.loads(data)

    def __repr__(self) -> str:
        r_str = f"{self.__class__.__name__} with:\n"
        r_str += "\n".join(f"\t{b}" for b in self.bands)
        if len(self.__stored_args__) != 0:
            r_str += "\nAnd the extra metadata:\n"
            for key, metadata in self.__stored_args__.items():
                r_str += f"\t{key}: {metadata}\n"
        return r_str
