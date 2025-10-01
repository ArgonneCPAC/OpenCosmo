from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, NamedTuple, Optional, Protocol, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from opencosmo.index import DataIndex, SimpleIndex
    from opencosmo.spatial.region import BoxRegion
    from opencosmo.units import UnitConvention
    from opencosmo.units.get import UnitApplicator

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
    def into_base_convention(
        self,
        converters: list["UnitApplicator"],
        columns: Iterable[str],
        from_: "UnitConvention",
        unit_kwargs: dict[str, Any],
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
    @property
    def subdivision_factor(self) -> int: ...
    def get_partition_region(self, index: SimpleIndex, level: int) -> Region:
        pass

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
