"""
I/O utilities for hdf5
"""

from typing import Optional

import h5py
import hdf5plugin  # type: ignore

from opencosmo.index import DataIndex


def write_index(
    input_ds: h5py.Dataset,
    output_group: h5py.Group,
    index: DataIndex,
    range_: Optional[tuple[int, int]] = None,
):
    if len(index) == 0:
        raise ValueError("No indices provided to write")
    data = index.get_data(input_ds)
    output_name = input_ds.name.split("/")[-1]
    compression = hdf5plugin.Blosc2(cname="lz4", filters=hdf5plugin.Blosc2.BITSHUFFLE)

    output_group.create_dataset(
        output_name, dtype=input_ds.dtype, data=data, compression=compression
    )
    attrs = input_ds.attrs
    for key in attrs.keys():
        output_group[output_name].attrs[key] = attrs[key]
