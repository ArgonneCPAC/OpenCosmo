from __future__ import annotations
from opencosmo.parameters import SimulationParameters
from opencosmo.spatial.region import BoxRegion, Point3d
from itertools import product
from typing import Iterable
from copy import copy

Index3d = tuple[int, int, int]


def get_index3d(p: Point3d, level: int, box_size: float) -> Index3d:
    block_size = box_size / (2 ** level)
    return int(p[0] // block_size), int(p[1] // block_size), int(p[2] // block_size)

def get_octtree_index(idx: Index3d, level: int, box_size: float) -> int:
    oct_idx = 0
    idx_ = copy(idx)
    for i in range(level):
        oct_idx |= (idx_[0] & 1) << 3*i
        oct_idx |= (idx_[1] & 1) << (3*i + 1)
        oct_idx |= (idx_[2] & 1) << (3*i + 2)
        idx_ = (idx_[0] >> 1, idx_[1] >> 1, idx_[2] >> 1)
    return oct_idx

def get_overlapping_blocks(region: BoxRegion, level: int, box_size: float) -> Iterable[Index3d]:
    p1 = get_index3d(region.p1, level, box_size)
    p2 = get_index3d(region.p2, level, box_size)
    i = range(p1[0], p2[0] + 1)
    j = range(p1[1], p2[1] + 1)
    k = range(p1[2], p2[2] + 1)
    return product(i, j, k)

class OctTreeIndex:
    def __init__(self, simulation_parameters: SimulationParameters, max_level: int):
        self.simulation_parameters = simulation_parameters
        self.max_level = max_level

    def get_regions(self, region: BoxRegion) -> Iterable[int]:
        # Deal with cases that are outside the simulation region
        blocks = get_overlapping_blocks(region, self.max_level, self.simulation_parameters.box_size)
        return [get_octtree_index(block, self.max_level, self.simulation_parameters.box_size) for block in blocks]



