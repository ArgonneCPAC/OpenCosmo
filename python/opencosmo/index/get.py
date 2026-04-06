from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np

from .unary import get_length

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_data(data: h5py.Dataset | np.ndarray, index: np.ndarray | tuple):
    if get_length(index) == 0:
        return np.array([])
    match index:
        case np.ndarray():
            return get_data_simple(data, index)
        case (np.ndarray(), np.ndarray()):
            return get_data_chunked(data, *index)
        case _:
            raise ValueError(f"Got invalid index of type {type(index)}")


def get_data_simple(data: h5py.Dataset | np.ndarray, index: NDArray[np.int_]):
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


def get_data_chunked(
    data: h5py.Dataset | np.ndarray, starts: NDArray[np.int_], sizes: NDArray[np.int_]
):
    """
    We assume that starts are ordered, and chunks are non-overlapping

    """

    unit = None
    if isinstance(data, u.Quantity):
        unit = data.unit

    shape = (np.sum(sizes),) + data.shape[1:]
    storage = np.zeros(shape, dtype=data.dtype)
    running_index = 0

    for i, (start, size) in enumerate(zip(starts, sizes)):
        source_slice = np.s_[start : start + size]
        dest_slice = np.s_[running_index : running_index + size]

        if isinstance(data, h5py.Dataset):
            data.read_direct(storage, source_slice, dest_slice)
        else:
            storage[dest_slice] = data[source_slice]

        running_index += size

    if unit is not None:
        storage *= unit
    return storage
