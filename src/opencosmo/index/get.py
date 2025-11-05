from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

import astropy.units as u
import h5py
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_data_simple(data: h5py.Dataset | np.ndarray, index: NDArray[np.int_]):
    if isinstance(data, np.ndarray):
        return data[index]

    min = index.min()
    max = index.max()
    remaining_shape = data.shape[1:]
    length = max + 1 - min

    shape = (length,) + remaining_shape

    buffer = np.zeros(shape, data.dtype)

    data.read_direct(buffer, np.s_[min : max + 1], np.s_[0:length])
    return buffer[index - min]


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
