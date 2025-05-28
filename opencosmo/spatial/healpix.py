import astropy.units as u  # type: ignore
import h5py
import numpy as np
from healpy import query_disc  # type: ignore
from healpy.pixelfunc import ang2vec  # type: ignore

from opencosmo.dataset.index import SimpleIndex
from opencosmo.spatial.protocols import Region
from opencosmo.spatial.region import ConeRegion


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

    def partition(self, n_partitions: int, max_level: int):
        raise NotImplementedError()

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
        if not isinstance(region, ConeRegion):
            raise ValueError("Didn't recieve a 2d region!")
        ra = region.center.ra.to(u.radian).value
        dec = region.center.dec.to(u.radian).value
        phi = ra  # SkyCoords casts negative RAs to their positive equivalent
        # declination of north pole is +pi / 2
        # Theta of north pole is 0
        theta = np.pi / 2 - dec

        vec = ang2vec(theta, phi)
        radius = region.radius.to(u.rad).value
        nside = 2**level
        intersects = query_disc(nside, vec, radius, inclusive=True, nest=True)
        return {level: (SimpleIndex.empty(), SimpleIndex(intersects))}
