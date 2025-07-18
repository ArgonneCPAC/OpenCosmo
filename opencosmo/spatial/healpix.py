import h5py
import numpy as np

from opencosmo.index import SimpleIndex
from opencosmo.spatial.protocols import Region
from opencosmo.spatial.region import HealPixRegion


class HealPixIndex:
    def __init__(self):
        pass

    @staticmethod
    def combine_upwards(counts: np.ndarray, level: int, target: h5py.File) -> h5py.File:
        if len(counts) != 12 * (4**level):
            raise ValueError("Recieved invalid number of counts!")
        group = target.require_group(f"level_{level}")
        new_starts = np.insert(np.cumsum(counts), 0, 0)[:-1]
        group.create_dataset("start", data=new_starts)
        group.create_dataset("size", data=counts)

        if level > 0:
            new_counts = counts.reshape(-1, 4).sum(axis=1)
            return HealPixIndex.combine_upwards(new_counts, level - 1, target)

        return target

    def get_partition_region(self, index: SimpleIndex, level: int) -> Region:
        idxs = index.into_array()
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
        return {level: (SimpleIndex.empty(), SimpleIndex(intersects))}
