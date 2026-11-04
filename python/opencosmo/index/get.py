from __future__ import annotations

import math
from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np

from opencosmo._lib import index as idxlib

from .mask import into_array
from .unary import get_length

MAX_SINGLE_READ = 8_000_000_000

if TYPE_CHECKING:
    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex


def get_data(data: h5py.Dataset | np.ndarray, index: DataIndex):
    if get_length(index) == 0:
        return np.array([], data.dtype)
    elif isinstance(data, np.ndarray):
        return data[into_array(index)]
    match index:
        case np.ndarray():
            return get_hdf5_data_simple(data, index)
        case (np.ndarray(), np.ndarray()):
            return get_hdf5_data_chunked(data, index)
        case _:
            raise ValueError(f"Got invalid index of type {type(index)}")


def __max_rows_per_read(data: h5py.Dataset) -> int:
    row_bytes = math.prod(data.shape[1:]) * data.dtype.itemsize
    return max(MAX_SINGLE_READ // row_bytes, 1)


def __read_runs(
    data: h5py.Dataset, run_starts: np.ndarray, run_sizes: np.ndarray
) -> np.ndarray:
    """Read ascending row runs from ``data`` into one contiguous buffer."""
    buffer = np.empty((int(run_sizes.sum()),) + data.shape[1:], dtype=data.dtype)
    offset = 0
    for start, size in zip(run_starts.tolist(), run_sizes.tolist(), strict=True):
        data.read_direct(
            buffer, np.s_[start : start + size], np.s_[offset : offset + size]
        )
        offset += size
    return buffer


def get_hdf5_data_simple(data: h5py.Dataset, index: SimpleIndex) -> np.ndarray:
    min_ = int(index.min())
    max_ = int(index.max())
    length = max_ + 1 - min_
    max_rows = __max_rows_per_read(data)

    if length <= max_rows:
        buffer = np.empty((length,) + data.shape[1:], data.dtype)
        data.read_direct(buffer, np.s_[min_ : max_ + 1], np.s_[0:length])
        return buffer[index - min_]

    # The span exceeds the read limit, so read nearby rows in bounded runs
    chunk_size = data.chunks[0] if data.chunks is not None else None
    run_starts, run_sizes, positions = idxlib.get_simple_read_order(
        index, chunk_size, max_rows
    )
    return __read_runs(data, run_starts, run_sizes)[positions]


def get_hdf5_data_chunked(data: h5py.Dataset, index: ChunkedIndex):
    """
    We do not assume that starts are ordered or non-overlapping

    """
    if len(index[0]) == 1:
        if get_length(index) == len(data):
            return data[:]
        start = index[0][0]
        size = index[1][0]
        return data[start : start + size]

    starts = index[0]
    sizes = index[1]

    unit = None
    if isinstance(data, u.Quantity):
        unit = data.unit

    chunk_size = data.chunks[0] if data.chunks is not None else None
    max_rows = __max_rows_per_read(data)

    new_start, new_size, new_output_start = idxlib.get_chunked_read_order(
        starts, sizes, chunk_size, max_rows
    )
    shape = (np.sum(new_size),) + data.shape[1:]
    storage = np.zeros(shape, dtype=data.dtype)
    rs = 0

    for (
        start,
        size,
    ) in zip(new_start, new_size, strict=True):
        source_slice = np.s_[start : start + size]
        dest_slice = np.s_[rs : rs + size]

        if isinstance(data, h5py.Dataset):
            data.read_direct(storage, source_slice, dest_slice)
        else:
            storage[dest_slice] = data[source_slice]
        rs += int(size)

    output_permutation = into_array((new_output_start, sizes))
    storage = storage[output_permutation]

    if unit is not None:
        storage *= unit
    return storage
