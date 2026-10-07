from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex
    from opencosmo.spatial.models import RegionModel
    from opencosmo.spatial.region import BoxRegion, HealpixRegion
    from opencosmo.spatial.types import SpatialIndexData
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

    def intersects(self, other: Region) -> bool: ...
    def contains(self, other: SpatialObject): ...
    def regularize(
        self,
        converters: list[UnitApplicator],
        columns: Iterable[str],
        from_: UnitConvention,
        unit_kwargs: dict[str, Any],
    ): ...
    def into_model(self) -> RegionModel: ...


@runtime_checkable
class Region2d(Region, Protocol):
    def get_healpix_intersections(self, nside: int): ...
    def into_healpix_region(self, nside: int) -> HealpixRegion: ...


@runtime_checkable
class Region3d(Region, Protocol):
    def bounding_box(self) -> BoxRegion: ...


class TreePartition(NamedTuple):
    idx: DataIndex
    region: Region | None
    level: int | None


class SpatialIndex(Protocol):
    @property
    def subdivision_factor(self) -> int: ...
    @property
    def level(self) -> int: ...
    @property
    def spatial_index_data(self) -> SpatialIndexData: ...
    def with_level(self, level: int) -> SpatialIndex: ...
    def get_partition_from_index(self, index: SimpleIndex) -> Region:
        pass

    def query(self, region: Region) -> tuple[ChunkedIndex, ChunkedIndex]:
        """
        Given a region in space, return a dictionary where each key is a level and each
        value is a tuple of DataIndexes. The first DataIndex corresponds to the regions
        that are fully contained by the given region, and the second corresponds to
        regions that only overlap.

        If a given subvolume is full contained by the query region, this method should
        NOT return any sub-sub volumes.
        """
        ...
