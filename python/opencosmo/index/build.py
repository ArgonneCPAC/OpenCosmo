from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mask import into_array

if TYPE_CHECKING:
    import h5py

    from . import ChunkedIndex, DataIndex, SimpleIndex


def from_size(size: int) -> ChunkedIndex:
    return (np.array([0], dtype=np.int64), np.array([size], dtype=np.int64))


def coalesce_chunks(starts: np.ndarray, sizes: np.ndarray) -> ChunkedIndex:
    """
    Merge consecutive chunks whose ranges are physically contiguous
    (starts[i] + sizes[i] == starts[i+1]) into single chunks. Preserves array
    order and the total row set, so the concatenated read output is
    byte-identical while the number of HDF5 reads is minimized.

    Only neighbors that are adjacent *in array order* are fused, so this is
    correct even when chunk order is deliberately permuted (resort/rebuild): it
    never reorders rows.
    """
    starts = starts.astype(np.int64)
    sizes = sizes.astype(np.int64)
    if len(starts) <= 1:
        return (starts, sizes)
    ends = starts[:-1] + sizes[:-1]
    new_group = np.concatenate(([True], starts[1:] != ends))
    group_idx = np.flatnonzero(new_group)
    new_starts = starts[group_idx]
    new_sizes = np.add.reduceat(sizes, group_idx)
    return (new_starts, new_sizes)


def single_chunk(start: int, size: int) -> ChunkedIndex:
    return (np.array([start], dtype=np.int64), np.array([size], np.int64))


def empty() -> ChunkedIndex:
    return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))


def zeros(length: int) -> SimpleIndex:
    return np.zeros(length, dtype=np.int64)


def from_range(start: int, end: int) -> ChunkedIndex:
    size = end - start
    return (np.array([start], dtype=np.int64), np.array([size], np.int64))


def concatenate(*indices: DataIndex) -> SimpleIndex:
    return np.concatenate(list(map(into_array, indices)))


def from_start_size_group(group: h5py) -> ChunkedIndex:
    start = group["start"][:].astype(np.int64)
    size = group["size"][:].astype(np.int64)
    return (start, size)
