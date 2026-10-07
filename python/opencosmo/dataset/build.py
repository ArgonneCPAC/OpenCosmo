from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, TypeVar

import h5py
import numpy as np

from opencosmo.dataset import Dataset, state
from opencosmo.spatial.index import HealpixIndex, get_region
from opencosmo.spatial.utils import combine_upwards

if TYPE_CHECKING:
    from astropy import units as u

    from opencosmo.header import OpenCosmoHeader

T = TypeVar("T")
GroupedColumnData = dict[str, dict[str, T]]
RawSpatialIndexData = dict[int, tuple[np.ndarray, int]]


def build_dataset_from_data(
    data: GroupedColumnData[np.ndarray],
    header: OpenCosmoHeader,
    spatial_index_data: RawSpatialIndexData | None,
    descriptions: GroupedColumnData[str] = {},
) -> Dataset:
    data_keys = set(data.keys())
    if data_keys != {"data"}:
        raise ValueError("Data must have exactly one `data` group")
    if descriptions and not set(descriptions.keys()).issubset(data.keys()):
        raise ValueError(
            "Descriptions should be organized into the same groups as the data!"
        )

    spatial_index = None
    spatial_index_columns = None
    if isinstance(spatial_index_data, dict):
        spatial_index_columns = make_spatial_index(spatial_index_data)
        spatial_index = HealpixIndex(max(spatial_index_data), spatial_index_columns)
    region = (
        get_region(spatial_index, None)
        if spatial_index is not None and spatial_index_columns is not None
        else None
    )
    data_group = data.pop("data")

    data_descriptions = descriptions.get("data", {})
    new_state = state.state_in_memory(
        data_group,
        header,
        header.file.unit_convention,
        {},
        data_descriptions,
        spatial_index=spatial_index,
        region=region,
    )
    return Dataset(new_state)


def build_dataset_from_evaluated_data(
    data: dict[str, np.ndarray | u.Quantity], header: OpenCosmoHeader
) -> Dataset:
    """Build an in-memory dataset from results in the file's unit convention."""
    return Dataset(state.state_from_evaluated_data(data, header))


def make_spatial_index(data: RawSpatialIndexData):
    """
    allowed input (for now)

    a single level > 0
    """
    if len(data) != 1:
        raise ValueError("Spatial index creation routines should have a single level")
    level = next(iter(data.keys()))
    size, fold_factor = data[level]
    if level <= 0:
        raise ValueError("Data for creating spatial index should include one level > 0")
    name = uuid.uuid1()
    file = h5py.File(f"{name}.hdf5", "w", driver="core", backing_store=False)
    data = combine_upwards(size, fold_factor, level, file)
    output = {}
    for group in data.values():
        assert isinstance(group, h5py.Group)
        output.update({ds.name[1:]: ds for ds in group.values()})
    return output
