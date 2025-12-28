from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import healpy as hp
import numpy as np

from opencosmo import dataset as ds
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.mpi import get_comm_world
from opencosmo.spatial.check import find_coordinates_2d

if TYPE_CHECKING:
    from opencosmo.io.schema import Schema


def update_order(data: np.ndarray, order: np.ndarray):
    return data[order]


def sync_headers(dataset: list[ds.Dataset], comm):
    pass


def stack_datasets_in_schema(datasets: list[ds.Dataset], name: str):
    if len(datasets) == 1:
        return datasets[0].make_schema(name=name)

    schemas = [ds.make_schema(name=name) for ds in datasets]
    reference_schema = schemas[0]
    index_names = list(schemas[0].children["index"].children.keys())
    index_names.sort()
    max_level = int(index_names[-1][-1])

    order = get_stacked_lightcone_order(datasets, max_level)
    assert all(isinstance(dataset, ds.Dataset) for dataset in datasets)
    new_data_group = stack_data_groups([schema.children["data"] for schema in schemas])

    new_index_group = stack_data_groups(
        [schema.children["index"] for schema in schemas]
    )
    if comm := get_comm_world() is not None:
        header_schema = sync_headers(datasets, comm)

    children = {"data": new_data_group, "index": new_index_group}

    return make_schema(name, FileEntry.DATASET, children=children)


def stack_data_groups(schemas: list[Schema]):
    if len(schemas) == 1:
        return schemas[0]
    base_schema = schemas[0]
    new_writers = {}
    for name, column_writer in base_schema.columns.items():
        other_writers = [schema.columns[name] for schema in schemas[1:]]
        new_writer = column_writer.concat(other_writers)
        new_writers[name] = new_writer

    new_schema = make_schema(
        base_schema.name,
        base_schema.type,
        children={},
        columns=new_writers,
        attributes=base_schema.attributes,
    )
    return new_schema


def get_stacked_lightcone_order(datasets: Iterable[ds.Dataset], max_index_depth: int):
    datasets = list(datasets)
    nside = 2**max_index_depth
    coordinates = list(map(find_coordinates_2d, datasets))
    pixels = np.concatenate(
        [
            hp.ang2pix(nside, coords.ra.value, coords.dec.value, lonlat=True, nest=True)
            for coords in coordinates
        ]
    )
    new_order = np.argsort(pixels)
    return np.split(new_order, [len(ds) for ds in datasets[:-1]])
