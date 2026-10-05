from __future__ import annotations

from typing import TYPE_CHECKING

from opencosmo._lib import spatial as spatlib
from opencosmo.spatial import builders

if TYPE_CHECKING:
    from opencosmo.index import SimpleIndex
    from opencosmo.spatial.protocols import Region3d

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


class OctTreeIndex:
    subdivision_factor = 8

    def __init__(self, box_size: float):
        """
        An octree index is used to spatialy index snapshot data.
        """
        self.box_size = box_size

    def get_partition_region(self, index: SimpleIndex, level: int):
        bounds = spatlib.partition_bounding_box(index, self.box_size, level)
        p1 = (bounds[0], bounds[2], bounds[4])
        p2 = (bounds[1], bounds[3], bounds[5])
        print(p1, p2)
        return builders.make_box(p1, p2)

    @classmethod
    def from_box_size(cls, box_size: int):
        return OctTreeIndex(box_size)

    def query(
        self, region: Region3d, max_level: int
    ) -> list[tuple[SimpleIndex, SimpleIndex]]:
        bbox = tuple(item for t in region.bounding_box().bounds for item in t)

        result = spatlib.get_octree_indices(self.box_size, bbox, max_level)
        return result
