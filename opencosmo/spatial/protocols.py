from typing import Protocol, Union

import h5py
import numpy as np
from astropy.cosmology import FLRW  # type: ignore
from numpy.typing import NDArray

from opencosmo.dataset.index import DataIndex, SimpleIndex
from opencosmo.spatial.region import BoxRegion
from opencosmo.transformations.units import UnitConvention

Point3d = tuple[float, float, float]
Point2d = tuple[float, float]
Points = NDArray[np.number]
SpatialObject = Union["Region", "Points"]


class Region(Protocol):
    """
    The region protocol is intentonally very vague, since we have to
    support both 2d regions and 3d regions.
    """

    def intersects(self, other: "Region") -> bool: ...
    def contains(self, other: SpatialObject): ...
    def into_scalefree(
        self, from_: UnitConvention, cosmology: FLRW, redshift: float
    ): ...


class Region2d(Region):
    def bounds(self): ...


class Region3d(Region, Protocol):
    def bounding_box(self) -> BoxRegion: ...


class SpatialIndex(Protocol):
    def partition(
        self, n_partitions: int, max_level: int
    ) -> tuple[list[DataIndex], int]: ...
    @staticmethod
    def combine_upwards(
        counts: np.ndarray, level: int, target: h5py.File
    ) -> h5py.File: ...
    def query(
        self, region: Region, max_level: int
    ) -> dict[int, tuple[SimpleIndex, SimpleIndex]]: ...
