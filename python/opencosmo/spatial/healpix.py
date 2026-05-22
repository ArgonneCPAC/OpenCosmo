from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from opencosmo.index import into_array
from opencosmo.spatial.region import HealpixRegion

if TYPE_CHECKING:
    from opencosmo.index import SimpleIndex
    from opencosmo.spatial.protocols import Region


class HealPixIndex:
    subdivision_factor = 4

    def __init__(self):
        pass

    def get_partition_region(self, index: SimpleIndex, level: int) -> Region:
        idxs = into_array(index)
        return HealpixRegion(idxs, 2**level)

    def query(
        self, region: Region, level: int = 1
    ) -> dict[int, tuple[SimpleIndex, SimpleIndex]]:
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
        if not hasattr(region, "get_healpix_intersections"):
            raise ValueError("Didn't recieve a 2D region!")
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
        return {
            level: (intersects[pixel_is_contained], intersects[~pixel_is_contained])
        }
