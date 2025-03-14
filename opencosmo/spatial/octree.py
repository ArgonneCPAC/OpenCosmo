from __future__ import annotations

from copy import copy
from functools import cache
from itertools import product
from typing import Iterable

from opencosmo.parameters import SimulationParameters
from opencosmo.spatial.region import Point3d

Index3d = tuple[int, int, int]


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
    def __init__(self, simulation_parameters: SimulationParameters, max_level: int):
        self.simulation_parameters = simulation_parameters
        self.max_level = max_level
