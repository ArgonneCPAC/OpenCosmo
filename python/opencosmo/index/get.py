from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np

from .unary import get_length

H5PY_USE_FANCY_INDEXING = h5py.version.version_tuple[:2] >= (3, 16)
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

    remaining_shape = data.shape[1:]
    row_size = data.dtype.itemsize * int(np.prod(remaining_shape))

    length = int(max_ + 1 - min_)
    frac_chunk_read = num_output_elements / length

    if isinstance(data, np.ndarray):
        return data[index]

    if (
        H5PY_USE_FANCY_INDEXING
        and num_output_elements < 10_000
        and frac_chunk_read < 0.01
    ):  # h5py 3.16 significantly improved fancy indexing for small selections
        # h5py requires strictly increasing indices
        unique, inverse = np.unique(index, return_inverse=True)
        return data[unique][inverse]

    total_out = row_size * num_output_elements
    total_read = row_size * length
    if total_read > SINGLE_READ_THRESHOLD and total_read > total_out * 2:
        return __get_data_blocked(data, index)

    shape = (length,) + remaining_shape

    buffer = np.zeros(shape, data.dtype)

    data.read_direct(buffer, np.s_[min_ : max_ + 1], np.s_[0:length])
    return buffer[index - min_]


def __get_data_blocked(data: h5py.Dataset, index: SimpleIndex):
    """
    Read a sparse selection as a series of contiguous spans, each at most
    SINGLE_READ_THRESHOLD bytes, scattering rows back into index order.
    """
    remaining_shape = data.shape[1:]
    row_size = data.dtype.itemsize * int(np.prod(remaining_shape))
    rows_per_read = max(1, SINGLE_READ_THRESHOLD // row_size)

    order = np.argsort(index, kind="stable")
    sorted_index = index[order]

    output = np.empty((len(index),) + remaining_shape, data.dtype)
    buffer_length = min(rows_per_read, int(sorted_index[-1] - sorted_index[0] + 1))
    buffer = np.empty((buffer_length,) + remaining_shape, data.dtype)

    block_start = 0
    while block_start < len(sorted_index):
        lo = int(sorted_index[block_start])
        block_end = int(np.searchsorted(sorted_index, lo + rows_per_read, side="left"))
        hi = int(sorted_index[block_end - 1]) + 1
        data.read_direct(buffer, np.s_[lo:hi], np.s_[0 : hi - lo])
        output[order[block_start:block_end]] = buffer[
            sorted_index[block_start:block_end] - lo
        ]
        block_start = block_end

    return output


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
