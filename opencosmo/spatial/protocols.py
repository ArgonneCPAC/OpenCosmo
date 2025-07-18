from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol, Union

import h5py
import numpy as np
from astropy.cosmology import FLRW  # type: ignore
from numpy.typing import NDArray

from opencosmo.index import DataIndex, SimpleIndex
from opencosmo.transformations.units import UnitConvention

if TYPE_CHECKING:
    from opencosmo.spatial.region import BoxRegion

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
        self,
        from_: UnitConvention,
        cosmology: FLRW,
        redshift: float | tuple[float, float],
    ): ...


class Region2d(Region):
    def bounds(self): ...
    def get_healpix_intersections(self, nside: int): ...


class Region3d(Region, Protocol):
    def bounding_box(self) -> "BoxRegion": ...


class TreePartition(NamedTuple):
    idx: DataIndex
    region: Optional[Region]
    level: Optional[int]


class SpatialIndex(Protocol):
    def get_partition_region(self, index: SimpleIndex, level: int) -> Region:
        pass

    @staticmethod
    def combine_upwards(counts: np.ndarray, level: int, target: h5py.File) -> h5py.File:
        """
        Given a count of the number of items in each region at a given level, write the
        starts and sizes to and hdf5 dataset and recursively work upwards until the
        top level of the index is reached. The index should verify that the length of
        the initial array it recieves is the correct length for the given level.
        """
        ...

    def query(
        self, region: Region, max_level: int
    ) -> dict[int, tuple[SimpleIndex, SimpleIndex]]:
        """
        Given a region in space, return a dictionary where each key is a level and each
        value is a tuple of DataIndexes. The first DataIndex corresponds to the regions
        that are fully contained by the given region, and the second corresponds to
        regions that only overlap.

        If a given subvolume is full contained by the query region, this method should
        NOT return any sub-sub volumes.
        """
        ...
