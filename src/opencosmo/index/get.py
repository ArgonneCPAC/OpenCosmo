import h5py
import numpy as np
from numpy.typing import NDArray


def pack(start: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine adjacent chunks into a single chunk.
    """

    # Calculate the end of each chunk
    end = start + size

    # Determine where a new chunk should start (i.e., not adjacent to previous)
    # We prepend True for the first chunk to always start a group
    new_group = np.ones(len(start), dtype=bool)
    new_group[1:] = start[1:] != end[:-1]

    # Assign a group ID for each segment
    group_ids = np.cumsum(new_group)

    # Combine chunks by group
    combined_start = np.zeros(group_ids[-1], dtype=start.dtype)
    combined_size = np.zeros_like(combined_start)

    np.add.at(combined_start, group_ids - 1, np.where(new_group, start, 0))
    np.add.at(combined_size, group_ids - 1, size)

    return combined_start, combined_size


def get_data_hdf5_simple(data: h5py.Dataset, index: NDArray[np.int_]):
    min = index.min()
    max = index.max()
    remaining_shape = data.shape[1:]
    length = max + 1 - min

    shape = (length,) + remaining_shape

    buffer = np.zeros(shape, data.dtype)

    data.read_direct(buffer, np.s_[min : max + 1], np.s_[0:length])
    return buffer[index - min]


def get_data_hdf5_chunked(
    data: h5py.Dataset, starts: NDArray[np.int_], sizes: NDArray[np.int_]
):
    """
    We assume that starts are ordered, and chunks are non-overlapping

    """

    shape = (np.sum(sizes),) + data.shape[1:]
    storage = np.zeros(shape, dtype=data.dtype)
    running_index = 0
    for i, (start, size) in enumerate(zip(starts, sizes)):
        data.read_direct(
            storage,
            np.s_[start : start + size],
            np.s_[running_index : running_index + size],
        )
        running_index += size
    return storage
