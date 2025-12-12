from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo.index import into_array
from opencosmo.spatial.region import HealPixRegion

if TYPE_CHECKING:
    from opencosmo.index import SimpleIndex
    from opencosmo.spatial.protocols import Region


class HealPixIndex:
    subdivision_factor = 4

    def __init__(self):
        pass

    def get_partition_region(self, index: SimpleIndex, level: int) -> Region:
        idxs = into_array(index)
        return HealPixRegion(idxs, 2**level)

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
        return {level: (np.array([]), intersects)}
