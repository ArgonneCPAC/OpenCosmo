from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid1

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from opencosmo._lib import spatial as spatlib
from opencosmo.index import from_size, get_data, into_array, n_in_range, project
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter, Hdf5Source
from opencosmo.spatial import builders
from opencosmo.spatial.protocols import Region2d, Region3d, TreePartition
from opencosmo.spatial.region import HealpixRegion
from opencosmo.spatial.utils import combine_upwards

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region, SpatialIndex
    from opencosmo.spatial.types import SpatialIndexData

Index3d = tuple[int, int, int]


"""
In an oct tree, the space is subdivided into octants. At level one, the space is 
subdivided into 8 octants with indexes (0, 0, 0) -> (1, 1, 1). At the next level, we 
have 64 octants labeled (0,0,0) -> (4,4,4) and so on.

To query, we traverse recursively. If the octant is completely enclosed by the query 
region, we simply return a version of that octant with no children. If the octant 
itersects the query region, we call the function on the octant's children. We then 
return a copy of an octant WITH the children that 

To evaluate the tree, we again traverse it recursively. If an octant has no children, 
we know all objects in that octant should be included in the output. Otherwise, we move 
on to the children.

However at the lowest level of the octant this breaks down. Here we instead get all the 
data for all of the octants, and check if they are contained by our query region.

"""


def build_data_index(
    indices: list[tuple[SimpleIndex, SimpleIndex]], spatial_index_data: SpatialIndexData
) -> tuple[ChunkedIndex, ChunkedIndex]:
    contains = []
    intersects = []
    for level, (contained, overlapping) in enumerate(indices):
        contains.append(_get_level_data(spatial_index_data, level, contained))
        intersects.append(_get_level_data(spatial_index_data, level, overlapping))
    return (
        np.concatenate([item[0] for item in contains]),
        np.concatenate([item[1] for item in contains]),
    ), (
        np.concatenate([item[0] for item in intersects]),
        np.concatenate([item[1] for item in intersects]),
    )


def query_healpix_partitions(
    region: Region, level: int
) -> list[tuple[SimpleIndex, SimpleIndex]]:
    assert isinstance(region, Region2d)
    nside = 2**level
    intersects = region.get_healpix_intersections(nside)
    boundaries = (
        hp.boundaries(nside, intersects, nest=True).transpose(0, 2, 1).reshape(-1, 3)
    )
    coords = SkyCoord(*hp.vec2ang(boundaries, lonlat=True), unit="deg")
    coord_is_contained = region.contains(coords)
    pixel_is_contained = np.all(coord_is_contained.reshape(-1, 4), axis=1)
    result = [
        (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)) for _ in range(level)
    ]
    result.append((intersects[pixel_is_contained], intersects[~pixel_is_contained]))
    return result


def _get_level_data(
    columns: SpatialIndexData, level: int, index: DataIndex | None = None
) -> tuple[h5py.Dataset | np.ndarray, h5py.Dataset | np.ndarray]:
    key = f"level_{level}"
    if key in columns:
        level_data = columns[key]
        start = level_data["start"]
        size = level_data["size"]
    else:
        start = columns[f"{key}/start"]
        size = columns[f"{key}/size"]
    if index is None:
        index = from_size(len(start))
    return get_data(start, index), get_data(size, index)


def apply_index(spatial_index: SpatialIndex, index: DataIndex) -> SpatialIndex:
    columns = spatial_index.spatial_index_data
    starts, sizes = _get_level_data(columns, spatial_index.level)
    n = n_in_range(index, starts, sizes)
    target = h5py.File(f"{uuid1()}.hdf5", "w", driver="core", backing_store=False)
    indexed_data = combine_upwards(
        n, spatial_index.subdivision_factor, spatial_index.level, target
    )
    if not isinstance(spatial_index, (OctTreeIndex, HealpixIndex)):
        raise TypeError(
            f"Unsupported spatial index type: {type(spatial_index).__name__}"
        )
    return replace(spatial_index, spatial_index_data=indexed_data)


def get_max_level(columns: SpatialIndexData) -> int:
    names = set(columns.keys())
    max_level = -1
    while f"level_{max_level + 1}" in names or f"level_{max_level + 1}/start" in names:
        max_level += 1
    if max_level == -1:
        raise ValueError("Tried to read a tree but no levels were found!")
    return max_level


def get_region(spatial_index: SpatialIndex, region: Region | None) -> Region:
    if region is not None:
        return region
    if not isinstance(spatial_index, HealpixIndex):
        raise RuntimeError("A snapshot spatial index requires an explicit region")
    pixels = get_partitions_with_data(spatial_index, spatial_index.level)
    return HealpixRegion(pixels, nside=2**spatial_index.level)


def apply_range_mask(
    mask: np.ndarray,
    range_: tuple[int, int],
    starts: dict[int, np.ndarray],
    sizes: dict[int, np.ndarray],
) -> dict[int, tuple[int, np.ndarray]]:
    """Given an index range, apply a same-sized mask to produce new sizes."""
    output_sizes = {}
    for level, st in starts.items():
        ends = st + sizes[level]
        overlaps_mask = ~((st > range_[1]) | (ends < range_[0]))
        first_start_index = int(np.argmax(overlaps_mask))
        st = st[overlaps_mask]
        st[0] = range_[0]
        st = st - range_[0]
        output_sizes[level] = (first_start_index, np.add.reduceat(mask, st))
    return output_sizes


def partition_index(
    n_partitions: int, counts: h5py.Group, min_level: int
) -> tuple[list[np.ndarray], int]:
    levels = [int(key.split("_")[1]) for key in counts.keys()]
    highest_level = max(levels)
    split_level = -1
    full_region_indices = np.empty(0, dtype=np.int64)
    for level in range(min_level, highest_level + 1):
        level_counts = counts[f"level_{level}"]["size"][:]
        full_region_indices = np.where(level_counts > 0)[0]
        n_full = len(full_region_indices)
        if n_full < n_partitions:
            continue
        if n_full % n_partitions == 0:
            split_level = level
            break
    if split_level == -1:
        split_level = highest_level
    return list(
        np.array_split(full_region_indices.astype(np.int64), n_partitions)
    ), split_level


def get_partitions_with_data(
    spatial_index: SpatialIndex,
    level: int,
    index: DataIndex | None = None,
) -> np.ndarray:
    columns = spatial_index.spatial_index_data
    if level > get_max_level(columns):
        raise ValueError("Requested level is greater than the max level of this tree!")
    starts, sizes = _get_level_data(columns, level)
    if index is None:
        return np.where(sizes > 0)[0]
    return np.searchsorted(starts, into_array(index), side="right") - 1


def project_on_index(
    spatial_index: SpatialIndex,
    level: int,
    index: DataIndex,
    partitions: DataIndex | None,
) -> DataIndex:
    columns = spatial_index.spatial_index_data
    if level > get_max_level(columns):
        raise ValueError(
            "Level must be less than or equal to the max level of this tree"
        )
    starts, sizes = _get_level_data(columns, level, partitions)
    return project(index, (starts, sizes))


def partition(
    spatial_index: SpatialIndex,
    n_partitions: int,
    counts: h5py.Group,
    min_level: int | None = None,
) -> Sequence[TreePartition]:
    partition_indices, split_level = partition_index(
        n_partitions, counts, min_level or 0
    )
    partitions = []
    for index in partition_indices:
        if len(index) == 0:
            continue
        index_starts, index_sizes = _get_level_data(
            spatial_index.spatial_index_data, split_level, index
        )
        idx = (
            np.atleast_1d(index_starts[0]),
            np.atleast_1d(np.sum(index_sizes)),
        )
        region = spatial_index.with_level(split_level).get_partition_from_index(index)
        partitions.append(TreePartition(idx, region, split_level))
    return partitions


def make_tree_schema(spatial_index: SpatialIndex) -> Schema:
    columns = spatial_index.spatial_index_data
    level_schemas = {}
    for level in range(get_max_level(columns) + 1):
        starts, sizes = _get_level_data(columns, level)
        index = from_size(len(starts))
        start_source = Hdf5Source(starts, index)
        size_source = Hdf5Source(sizes, index)
        level_schemas[f"level_{level}"] = make_schema(
            f"level_{level}",
            FileEntry.COLUMNS,
            columns={
                "size": ColumnWriter([size_source], ColumnCombineStrategy.SUM),
                "start": ColumnWriter([start_source], ColumnCombineStrategy.SUM),
            },
        )
    return make_schema("index", FileEntry.COLUMNS, children=level_schemas)


@dataclass(frozen=True)
class OctTreeIndex:
    subdivision_factor: ClassVar[int] = 8
    box_size: float
    level: int
    spatial_index_data: SpatialIndexData

    def with_level(self, level: int) -> OctTreeIndex:
        return OctTreeIndex(self.box_size, level, self.spatial_index_data)

    def get_partition_from_index(self, index: SimpleIndex):
        bounds = spatlib.partition_bounding_box(index, self.box_size, self.level)
        p1 = (bounds[0], bounds[2], bounds[4])
        p2 = (bounds[1], bounds[3], bounds[5])
        return builders.make_box(p1, p2)

    def query(
        self,
        region: Region,
    ) -> tuple[ChunkedIndex, ChunkedIndex]:
        assert isinstance(region, Region3d)
        bbox = tuple(item for t in region.bounding_box().bounds for item in t)

        result = spatlib.get_octree_indices(self.box_size, bbox, self.level)
        return build_data_index(result, self.spatial_index_data)


@dataclass(frozen=True)
class HealpixIndex:
    subdivision_factor: ClassVar[int] = 4
    level: int
    spatial_index_data: SpatialIndexData

    def with_level(self, level: int) -> HealpixIndex:
        return HealpixIndex(level, self.spatial_index_data)

    def get_partition_from_index(self, index: SimpleIndex) -> HealpixRegion:
        idxs = into_array(index)
        return HealpixRegion(idxs, 2**self.level)

    def query(self, region: Region) -> tuple[ChunkedIndex, ChunkedIndex]:
        """
        Raw healpix data is

        - pi < phi < pi
        0 < theta < pi

        SkyCoordinates are typically

        0 < RA < 360 deg
        - 90 deg < Dec < 90 deg

        And HealPix is

        0 < phi < 2*pi
        0 < theta < pi

        This is why we can't have nice things
        """
        result = query_healpix_partitions(region, self.level)
        return build_data_index(result, self.spatial_index_data)


def open_spatial_index(
    columns: SpatialIndexData,
    box_size: float | None,
    is_lightcone: bool = False,
) -> SpatialIndex:
    level = get_max_level(columns)
    if is_lightcone:
        return HealpixIndex(level, columns)
    if box_size is None:
        raise ValueError("Cannot open a snapshot spatial index without a box size")
    return OctTreeIndex(box_size, level, columns)
