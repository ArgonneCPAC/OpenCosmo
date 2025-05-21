from __future__ import annotations

from collections import defaultdict
from copy import copy
from enum import Enum
from functools import cache
from itertools import product
from typing import Iterable, Optional, TypeGuard

import numpy as np

import opencosmo as oc
from opencosmo.dataset.index import SimpleIndex
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


def get_children(idx: Index3d) -> Iterable[Index3d]:
    return (
        (idx[0] * 2 + dx, idx[1] * 2 + dy, idx[2] * 2 + dz)
        for dx, dy, dz in product(range(2), repeat=3)
    )


class OctTreeIndex:
    def __init__(self, root: Octant):
        self.root = root

    @classmethod
    def from_box_size(cls, box_size: int):
        halfwidth = box_size / 2
        root = Octant((0, 0, 0), (halfwidth, halfwidth, halfwidth), halfwidth)
        return OctTreeIndex(root)

    def partition(self, n: int):
        level = 0
        n_ = n
        while n_ != 1:
            level += 1
            n_ = int(n_ / 8) if n_ > 8 else 1
        partitions = self.root.partition(n)
        partition_indices = [
            SimpleIndex(np.array([get_octtree_index(r.idx, level) for r in p]))
            for p in partitions
        ]
        return partition_indices, level

    def query(
        self, region: BoxRegion, max_level: int
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

    def partition(self, n: int) -> list[list[Octant]]:
        # Note, n is guaranteed to be a power of 2 at this point
        self.make_children()
        if n == 1:
            return [[self]]
        elif n == 2 or n == 4:
            return [self.children[n * i : n * (i + 1)] for i in range(8 // n)]
        elif n == 8:
            return [[child] for child in self.children]

        subidivisons_per_octant = n // 8
        partitions = []
        for child in self.children:
            partitions += child.partition(subidivisons_per_octant)

        return partitions

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
        return oc.Box(self.center, 2 * self.halfwidth)

    def query(
        self,
        region: BoxRegion,
        current_level: int,
        max_level: int,
        containment: dict[Octant, Intersection],
    ) -> Optional[Octant]:
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
