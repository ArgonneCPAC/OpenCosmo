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


def get_hdf5_data_simple(data: h5py.Dataset, index: SimpleIndex):
    if isinstance(data, np.ndarray):
        return data[index]

    min_ = index.min()
    max_ = index.max()
    remaining_shape = data.shape[1:]
    length = int(max_ + 1 - min_)

    shape = (length,) + remaining_shape

    buffer = np.zeros(shape, data.dtype)

    data.read_direct(buffer, np.s_[min_ : max_ + 1], np.s_[0:length])
    return buffer[index - min_]


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
    elems_per_row = math.prod(data.shape[1:])
    max_elems_read = MAX_SINGLE_READ // (elems_per_row * data.dtype.itemsize)

    assert isinstance(chunk_size, int)
    new_start, new_size, new_output_start = idxlib.get_chunked_read_order(
        starts, sizes, chunk_size, max_elems_read
    )
    shape = (np.sum(new_size),) + data.shape[1:]
    storage = np.zeros(shape, dtype=data.dtype)
    rs = 0

    for (
        start,
        size,
    ) in zip(new_start, new_size):
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
