"""
I/O utilities for hdf5
"""

from typing import Optional

import hdf5plugin  # type: ignore
import numpy as np
from astropy.table import Column  # type: ignore
from h5py import Dataset, Group


def read_indices(
    ds: Dataset, indices: np.ndarray, range_: Optional[tuple[int, int]] = None
) -> Column:
    if len(indices) == 0:
        return Column([], name=ds.name)
    indices_into_data = indices
    if range_ is not None:
        indices_into_data = indices_into_data + range_[0]
    else:
        range_ = (0, indices_into_data.max())

    if indices_into_data.max() > range_[1]:
        raise ValueError("Tried to get indices outside the range of the dataset")

    data = ds[range_[0] : range_[1] + 1]
    return Column(data[indices_into_data - range_[0]], name=ds.name)


def write_indices(
    input_ds: Dataset,
    output_group: Group,
    indices: np.ndarray,
    range_: Optional[tuple[int, int]] = None,
):
    if len(indices) == 0:
        raise ValueError("No indices provided to write")
    data = read_indices(input_ds, indices, range_).data
    output_name = input_ds.name.split("/")[-1]
    compression = hdf5plugin.Blosc2(cname="lz4", filters=hdf5plugin.Blosc2.BITSHUFFLE)

    output_group.create_dataset(
        output_name, dtype=input_ds.dtype, data=data, compression=compression
    )
    attrs = input_ds.attrs
    for key in attrs.keys():
        output_group[output_name].attrs[key] = attrs[key]
