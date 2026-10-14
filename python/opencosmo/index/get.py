from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np

from .unary import get_length

H5PY_VERSION = tuple(int(v) for v in h5py.__version__.split("."))
H5PY_USE_FANCY_INDEXING = all(a >= b for a, b in zip(H5PY_VERSION, (3, 16)))
SINGLE_READ_THRESHOLD = 8_000_000_000

if TYPE_CHECKING:
    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex


def get_data(data: h5py.Dataset | np.ndarray, index: DataIndex):
    if get_length(index) == 0:
        return np.array([], data.dtype)
    match index:
        case np.ndarray():
            return get_data_simple(data, index)
        case (np.ndarray(), np.ndarray()):
            return get_data_chunked(data, index)
        case _:
            raise ValueError(f"Got invalid index of type {type(index)}")


def get_data_simple(data: h5py.Dataset | np.ndarray, index: SimpleIndex):
    num_output_elements = len(index)
    min_ = index.min()
    max_ = index.max()

    dtype_size = data.dtype.itemsize

    chunk_size = max_ - min_ + 1
    frac_chunk_read = len(index) / chunk_size

    if isinstance(data, np.ndarray) or (
        H5PY_USE_FANCY_INDEXING
        and num_output_elements < 10_000
        and frac_chunk_read < 0.01
    ):  # h5py 3.16 significantly improved fancy indexing for small selections
        return data[index]

    total_out = dtype_size * len(index)
    total_read = dtype_size * chunk_size
    if total_read > SINGLE_READ_THRESHOLD and total_read > total_out * 2:
        pass

    remaining_shape = data.shape[1:]
    length = int(max_ + 1 - min_)

    shape = (length,) + remaining_shape

    buffer = np.zeros(shape, data.dtype)

    data.read_direct(buffer, np.s_[min_ : max_ + 1], np.s_[0:length])
    return buffer[index - min_]


def get_data_chunked(data: h5py.Dataset | np.ndarray, index: ChunkedIndex):
    """
    We assume that starts are ordered, and chunks are non-overlapping

    """

    starts = index[0]
    sizes = index[1]

    unit = None
    if isinstance(data, u.Quantity):
        unit = data.unit

    shape = (np.sum(sizes),) + data.shape[1:]
    storage = np.zeros(shape, dtype=data.dtype)
    running_index = 0

    for _, (start, size) in enumerate(zip(starts, sizes)):
        source_slice = np.s_[start : start + size]
        dest_slice = np.s_[running_index : running_index + size]

        if isinstance(data, h5py.Dataset):
            data.read_direct(storage, source_slice, dest_slice)
        else:
            storage[dest_slice] = data[source_slice]

        running_index += int(size)

    if unit is not None:
        storage *= unit
    return storage
