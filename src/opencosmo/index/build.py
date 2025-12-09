from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mask import into_array

if TYPE_CHECKING:
    from . import DataIndex


def from_size(size: int):
    return (np.atleast_1d(0), np.atleast_1d(size))


def single_chunk(start: int, size: int):
    return (np.atleast_1d(start), np.atleast_1d(size))


def empty():
    return (np.array([], dtype=int), np.array([], dtype=int))


def concatenate(*indices: DataIndex):
    np.concatenate(list(map(into_array, indices)))
