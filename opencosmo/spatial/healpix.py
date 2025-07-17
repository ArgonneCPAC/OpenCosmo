from itertools import repeat

import h5py
import numpy as np

from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex
from opencosmo.spatial.protocols import Region, TreePartition
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

    def partition(
        self, n_partitions: int, max_level: int, counts: h5py.Group
    ) -> list[TreePartition]:
        level = 0
        n_per = 0.0
        level_counts: np.ndarray
        for i in range(max_level + 1):
            level_counts = counts[f"level_{i}"]["size"][:]

            n_cells = np.sum(level_counts > 0)
            if n_cells < n_partitions:
                continue

            n_per = n_cells / n_partitions
            partitions = np.split(np.where(level_counts > 0)[0], n_partitions)
            level = i
            break
        if n_per == 0.0:
            # The lowest level has less full cells than the
            # requested number of partitions
            total = np.sum(level_counts)
            nonzero = np.where(level_counts > 0)[0]
            nper = total // n_partitions
            starts = [nper * i for i in range(n_partitions)]
            sizes = [nper for _ in range(n_partitions)]
            sizes[-1] = total - starts[-1]
            indices = map(
                lambda arg: ChunkedIndex.single_chunk(arg[0], arg[1]),
                zip(starts, sizes),
            )
            regions = repeat(HealPixRegion(nonzero, 2**max_level))
            return list(
                map(
                    lambda arg: TreePartition(arg[0], arg[1], max_level),
                    zip(indices, regions),
                )
            )

        if not n_per.is_integer():
            raise NotImplementedError()

        level_starts = counts[f"level_{level}"]["start"][:]
        final_indices: list[DataIndex] = []
        final_regions: list[Region] = []

        for part in partitions:
            start = level_starts[part[0]]
            size = np.sum(level_counts[part])
            region = HealPixRegion(part, 2**level)
            index = ChunkedIndex.single_chunk(start, size)
            final_regions.append(region)
            final_indices.append(index)

        output_partitions = map(
            lambda in_: TreePartition(in_[0], in_[1], level),
            zip(final_indices, final_regions),
        )
        return list(output_partitions)

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
