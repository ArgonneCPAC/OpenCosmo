from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from opencosmo._lib import spatial as spatlib
from opencosmo.index import into_array
from opencosmo.spatial import builders
from opencosmo.spatial.protocols import Region2d, Region3d
from opencosmo.spatial.region import HealpixRegion

if TYPE_CHECKING:
    from opencosmo.index import SimpleIndex
    from opencosmo.spatial.protocols import Region

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


@dataclass(frozen=True)
class OctTreeIndex:
    subdivision_factor: ClassVar[int] = 8
    box_size: float

    def get_partition_from_index(self, index: SimpleIndex, level: int):
        bounds = spatlib.partition_bounding_box(index, self.box_size, level)
        p1 = (bounds[0], bounds[2], bounds[4])
        p2 = (bounds[1], bounds[3], bounds[5])
        return builders.make_box(p1, p2)

    def query(
        self, region: Region, level: int
    ) -> list[tuple[SimpleIndex, SimpleIndex]]:
        assert isinstance(region, Region3d)
        bbox = tuple(item for t in region.bounding_box().bounds for item in t)

        result = spatlib.get_octree_indices(self.box_size, bbox, level)
        return result


@dataclass(frozen=True)
class HealpixIndex:
    subdivision_factor = 4

    def get_partition_from_index(self, index: SimpleIndex, level: int) -> HealpixRegion:
        idxs = into_array(index)
        return HealpixRegion(idxs, 2**level)

    def query(
        self, region: Region, level: int = 1
    ) -> list[tuple[SimpleIndex, SimpleIndex]]:
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
        print(region)
        assert isinstance(region, Region2d)
        nside = 2**level
        intersects = region.get_healpix_intersections(nside)
        boundaries = (
            hp.boundaries(nside, intersects, nest=True)
            .transpose(
                0,
                2,
                1,
            )
            .reshape(-1, 3)
        )
        coords = SkyCoord(*hp.vec2ang(boundaries, lonlat=True), unit="deg")
        coord_is_contained = region.contains(coords)
        pixel_is_contained = np.all(coord_is_contained.reshape(-1, 4), axis=1)
        result = [
            (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            for _ in range(level)
        ]

        result.append((intersects[pixel_is_contained], intersects[~pixel_is_contained]))
        return result
