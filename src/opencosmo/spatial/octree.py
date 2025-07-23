from __future__ import annotations

from collections import defaultdict
from copy import copy
from enum import Enum
from functools import cache
from itertools import product
from typing import Iterable, Optional, TypeGuard

import h5py
import numpy as np

import opencosmo as oc
from opencosmo.index import SimpleIndex
from opencosmo.spatial.protocols import Region
from opencosmo.spatial.region import BoxRegion, Point3d

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


@cache
def get_index3d(p: Point3d, level: int, box_size: float) -> Index3d:
    block_size = box_size / (2**level)
    return int(p[0] // block_size), int(p[1] // block_size), int(p[2] // block_size)


def get_octtree_index(idx: Index3d, level: int) -> int:
    oct_idx = 0
    idx_ = copy(idx)
    for i in range(level):
        oct_idx |= (idx_[0] & 1) << 3 * i
        oct_idx |= (idx_[1] & 1) << (3 * i + 1)
        oct_idx |= (idx_[2] & 1) << (3 * i + 2)
        idx_ = (idx_[0] >> 1, idx_[1] >> 1, idx_[2] >> 1)
    return oct_idx


def get_3d_index(oct_idx: int, level: int) -> tuple:
    """
    Inverts the z-order curve indexing to retrieve the 3D index (i, j, k) from the
    1D index.

    Args:
        oct_idx (int): The 1D index obtained from the z-order curve.
        level (int): The depth level of the OctTree.

    Returns:
        tuple: The 3D index (i, j, k).
    """
    i, j, k = 0, 0, 0  # Initialize the 3D indices
    for le in range(level):
        # Extract the bits for i, j, and k at the current level
        i |= ((oct_idx >> (3 * le)) & 1) << le
        j |= ((oct_idx >> (3 * le + 1)) & 1) << le
        k |= ((oct_idx >> (3 * le + 2)) & 1) << le
    return i, j, k


def get_octant(oct_idx: int, level: int, box_size: float) -> Octant:
    index3d = get_3d_index(oct_idx, level)
    nside = 2**level
    octant_size = box_size / nside
    center = tuple(octant_size * (i + 0.5) for i in index3d)
    return Octant(index3d, center, octant_size / 2)


def get_children(idx: Index3d) -> Iterable[Index3d]:
    return (
        (idx[0] * 2 + dx, idx[1] * 2 + dy, idx[2] * 2 + dz)
        for dx, dy, dz in product(range(2), repeat=3)
    )


def get_region(octants: list[Octant]) -> BoxRegion:
    """
    Return the region that encloses this list of octants
    """
    if len(octants) == 1:
        return octants[0].bounding_box()
    bounds = map(lambda oct: np.array(oct.bounding_box().bounds), octants)
    bound_arr = np.stack(list(bounds))

    mins = bound_arr[:, :, 0].min(axis=0)
    maxs = bound_arr[:, :, 1].max(axis=0)
    return oc.make_box(tuple(mins), tuple(maxs))


class OctTreeIndex:
    def __init__(self, root: Octant):
        """
        An octree index is used to spatialy index snapshot data.
        """
        self.root = root

    @staticmethod
    def combine_upwards(counts: np.ndarray, level: int, target: h5py.File) -> h5py.File:
        if len(counts) != 8**level:
            raise ValueError("Recieved invalid number of counts!")
        group = target.require_group(f"level_{level}")
        new_starts = np.insert(np.cumsum(counts), 0, 0)[:-1]
        group.create_dataset("start", data=new_starts)
        group.create_dataset("size", data=counts)

        if level > 0:
            new_counts = counts.reshape(-1, 8).sum(axis=1)
            return OctTreeIndex.combine_upwards(new_counts, level - 1, target)

        return target

    def get_partition_region(self, index: SimpleIndex, level: int):
        octants = [
            get_octant(idx, level, 2 * self.root.halfwidth)
            for idx in index.into_array()
        ]
        return get_region(octants)

    @classmethod
    def from_box_size(cls, box_size: int):
        halfwidth = box_size / 2
        root = Octant((0, 0, 0), (halfwidth, halfwidth, halfwidth), halfwidth)
        return OctTreeIndex(root)

    def query(
        self, region: Region, max_level: int
    ) -> dict[int, tuple[SimpleIndex, SimpleIndex]]:
        containment: dict[Octant, Intersection] = {}
        new_root = self.root.query(region, 0, max_level, containment)
        if new_root is None:
            raise ValueError("Query region is not within the actual simulation box!")
        output = {}
        for level, (cidx, iidx) in make_octree_indices(
            new_root, 0, containment
        ).items():
            output[level] = (SimpleIndex(cidx), SimpleIndex(iidx))

        return output


def make_octree_indices(
    oct: Octant, current_level: int, containment: dict[Octant, Intersection]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Returns two arrays per level. One for regions that are fully contained.
    The other for regions that only intersect.
    """
    oct_intersection = containment[oct]
    current_level_output = (np.array([], dtype=int), np.array([], dtype=int))
    idx = get_octtree_index(oct.idx, current_level)
    output: dict[int, tuple[np.ndarray, np.ndarray]] = defaultdict(
        lambda: (np.array([], dtype=int), np.array([], dtype=int))
    )
    if oct_intersection == Intersection.CONTAINED:
        current_level_output = (np.array([idx], dtype=int), np.array([], dtype=int))
        output[current_level] = current_level_output

    elif oct_intersection == Intersection.INTERSECTS and len(oct.children) == 0:
        current_level_output = (np.array([], dtype=int), np.array([idx], dtype=int))
        output[current_level] = current_level_output

    elif oct_intersection == Intersection.INTERSECTS:
        for child in oct.children:
            child_output = make_octree_indices(child, current_level + 1, containment)
            for key, indices in child_output.items():
                current_output = output[key]
                new_contains = np.append(current_output[0], indices[0])
                new_intersects = np.append(current_output[1], indices[1])
                output[key] = (new_contains, new_intersects)

    return output


class Intersection(Enum):
    NONE = 0
    INTERSECTS = 1
    CONTAINED = 2


class Octant:
    def __init__(
        self,
        idx: Index3d,
        center: Point3d,
        halfwidth: float,
        children: Optional[list[Octant]] = None,
    ):
        self.idx = idx
        self.center = center
        self.halfwidth = halfwidth
        self.children = children if children is not None else []

    def __repr__(self):
        return f"{self.idx}: {self.bounding_box()}"

    def __hash__(self):
        return hash((self.idx, self.center, self.halfwidth))

    def __eq__(self, other):
        if not isinstance(other, Octant):
            return False

        output = (
            self.idx == other.idx
            and self.center == other.center
            and self.halfwidth == other.halfwidth
        )
        return output

    def make_children(self):
        if len(self.children) == 8:
            return

        child_halfwidth = self.halfwidth / 2.0
        for z, y, x in product(range(2), repeat=3):
            child_idx = (self.idx[0] * 2 + x, self.idx[1] * 2 + y, self.idx[2] * 2 + z)
            child_center = (
                self.center[0] + child_halfwidth * (2 * x - 1),
                self.center[1] + child_halfwidth * (2 * y - 1),
                self.center[2] + child_halfwidth * (2 * z - 1),
            )
            child = Octant(child_idx, child_center, child_halfwidth)
            self.children.append(child)

    @cache
    def bounding_box(self):
        p1 = tuple(ci - self.halfwidth for ci in self.center)
        p2 = tuple(ci + self.halfwidth for ci in self.center)

        return oc.make_box(p1, p2)

    def query(
        self,
        region: Region,
        current_level: int,
        max_level: int,
        containment: dict[Octant, Intersection],
    ) -> Optional[Octant]:
        if not isinstance(region, BoxRegion):
            raise ValueError("Did not recieve a 3D region!")
        if region.contains(self.bounding_box()):
            containment[self] = Intersection.CONTAINED
            return Octant(self.idx, self.center, self.halfwidth)
        if region.intersects(self.bounding_box()):
            containment[self] = Intersection.INTERSECTS
            if current_level == max_level:
                return Octant(self.idx, self.center, self.halfwidth)

            self.make_children()
            queried_children = map(
                lambda reg: reg.query(
                    region, current_level + 1, max_level, containment
                ),
                self.children,
            )

            def is_not_none(reg: Octant | None) -> TypeGuard[Octant]:
                return reg is not None

            new_children: list[Octant] = list(filter(is_not_none, queried_children))
            return Octant(self.idx, self.center, self.halfwidth, new_children)

        containment[self] = Intersection.NONE
        return None
