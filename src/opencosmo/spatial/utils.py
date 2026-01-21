from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py


def combine_upwards(
    counts: np.ndarray, factor: int, level: int, target: h5py.File
) -> h5py.File:
    """
    Given a count of the number of items in each region at a given level, write the
    starts and sizes to and hdf5 dataset and recursively work upwards until the
    top level of the index is reached. The index should verify that the length of
    the initial array it recieves is the correct length for the given level.
    """
    if (len(counts) % factor**level) != 0:
        raise ValueError("Recieved invalid number of counts!")

    group = target.require_group(f"level_{level}")
    new_starts = np.insert(np.cumsum(counts, dtype=np.int32), 0, 0)[:-1]
    counts = counts.astype(np.int32)  # This should be fixed
    group.create_dataset("start", data=new_starts)
    group.create_dataset("size", data=counts)

    if level > 0:
        new_counts = counts.reshape(-1, factor).sum(axis=1)
        return combine_upwards(new_counts, factor, level - 1, target)

    return target
