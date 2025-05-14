from __future__ import annotations

from copy import copy
from functools import cache
from itertools import product
from typing import Iterable, Optional, TypeGuard

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


def get_octtree_index(idx: Index3d, level: int, box_size: float) -> int:
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

    def query(self, reg: BoxRegion):
        pass


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

    def make_children(self):
        if len(self.children) != 0:
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
        return BoxRegion(self.center, self.halfwidth)

    def query(
        self, region: BoxRegion, current_level: int, max_level: int
    ) -> Optional[Octant]:
        if region.contains(self.bounding_box()) | (
            region.intersects(self.bounding_box()) and current_level == max_level
        ):
            return Octant(self.idx, self.center, self.halfwidth)
        if region.intersects(self.bounding_box()):
            self.make_children()
            queried_children = map(
                lambda reg: reg.query(region, current_level + 1, max_level),
                self.children,
            )

            def is_not_none(reg: Octant | None) -> TypeGuard[Octant]:
                return reg is not None

            new_children: list[Octant] = list(filter(is_not_none, queried_children))
            return Octant(self.idx, self.center, self.halfwidth, new_children)

        return None
