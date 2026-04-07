from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mask import into_array

if TYPE_CHECKING:
    import h5py

    from . import ChunkedIndex, DataIndex, SimpleIndex


def from_size(size: int) -> ChunkedIndex:
    return (np.array([0], dtype=np.int64), np.array([size], dtype=np.int64))


def single_chunk(start: int, size: int) -> ChunkedIndex:
    return (np.array([start], dtype=np.int64), np.array([size], np.int64))


def empty() -> ChunkedIndex:
    return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))


def from_range(start: int, end: int) -> ChunkedIndex:
    size = end - start
    return (np.array([start], dtype=np.int64), np.array([size], np.int64))


def concatenate(*indices: DataIndex) -> SimpleIndex:
    return np.concatenate(list(map(into_array, indices)))


def from_start_size_group(group: h5py) -> ChunkedIndex:
    start = group["start"][:].astype(np.int64)
    size = group["size"][:].astype(np.int64)
    return (start, size)
