from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mask import into_array

if TYPE_CHECKING:
    from . import DataIndex


def from_size(size: int):
    return (np.array([0], dtype=np.int64), np.array([size], dtype=np.int64))


def single_chunk(start: int, size: int):
    return (np.array([start], dtype=np.int64), np.array([size], np.int64))


def empty():
    return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))


def from_range(start: int, end: int):
    size = end - start
    return (np.array([start], dtype=np.int64), np.array([size], np.int64))


def concatenate(*indices: DataIndex):
    np.concatenate(list(map(into_array, indices)))
