from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, TypeVar

import astropy.units as u
import h5py

from opencosmo.dataset import Dataset
from opencosmo.dataset.state import DatasetState

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.spatial import Region

T = TypeVar("T")
GroupedColumnData = dict[str, dict[str, T]]


def build_dataset_from_data(
    data: GroupedColumnData[np.ndarray] | h5py.Group,
    header: OpenCosmoHeader,
    region: Region,
    spatial_index_data: Optional[GroupedColumnData | h5py.Group] = None,
    descriptions: GroupedColumnData[str] = {},
) -> Dataset:
    data_keys = set(data.keys())
    metadata_group = None
    if not data_keys or len(data_keys) > 2 or "data" not in data_keys:
        raise ValueError(
            "Data must have at least one `data` group and at most one metadata group"
        )
    if descriptions and not set(descriptions.keys()).issubset(data.keys()):
        raise ValueError(
            "Descriptions should be organized into the same groups as the data!"
        )

    if isinstance(data, dict):
        data = make_in_memory_h5_file_from_data(data, descriptions)
    if isinstance(spatial_index_data, dict):
        spatial_index_data = make_in_memory_h5_file_from_data(spatial_index_data)
    if len(data_keys) == 2:
        data_keys.remove("data")
        metadata_key = data_keys.pop()
        metadata_group = data[metadata_key]

    new_state = DatasetState.from_group(
        data["data"],
        header,
        header.file.unit_convention,
        region,
        metadata_group=metadata_group,
    )
    return Dataset(header, new_state)


def make_in_memory_h5_file_from_data(
    data: GroupedColumnData[np.ndarray],
    descriptions: Optional[GroupedColumnData[str]] = None,
) -> h5py.File:
    name = uuid.uuid1()
    file = h5py.File(f"{name}.hdf5", "w", driver="core", backing_store=False)
    for group_name, columns in data.items():
        lengths = set(len(c) for c in columns.values())
        if len(lengths) > 1:
            raise ValueError("Columns within a single group must be the same length!")
        group = file.require_group(group_name)
        group_data, group_metadata = split_data_and_metadata(
            columns,
            descriptions.get(group_name, {}) if descriptions is not None else {},
        )

        for column_name, column_data in columns.items():
            group.create_dataset(column_name, data=column_data)
            group[column_name].attrs.update(group_metadata[column_name])
    return file


def split_data_and_metadata(data: dict[str, np.ndarray], descriptions: dict[str, str]):
    output_data = {}
    output_metadata = {}
    for colname, coldata in data.items():
        output_data[colname] = coldata
        column_metadata = {}
        if isinstance(coldata, u.Quantity):
            column_metadata["unit"] = str(coldata.unit)
            output_data[colname] = coldata.value
        if colname in descriptions:
            column_metadata["description"] = descriptions[colname]
        output_metadata[colname] = column_metadata
    return output_data, output_metadata
