from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Iterable, Optional

import healpy as hp
import numpy as np

from opencosmo import dataset as ds
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.mpi import get_comm_world
from opencosmo.spatial.check import find_coordinates_2d

if TYPE_CHECKING:
    from opencosmo.io.schema import Schema
    from opencosmo.mpi import MPI


def update_order(data: np.ndarray, comm: Optional[MPI.Comm], order: np.ndarray):
    return data[order]


def sync_headers(datasets: list[ds.Dataset], redshift_range):
    steps = (
        dataset.header.file.step
        for dataset in datasets
        if dataset.header.file.step is not None
    )
    redshifts = (
        dataset.header.file.redshift
        for dataset in datasets
        if dataset.header.file.redshift is not None
    )
    step = max(steps)
    redshift = max(redshifts)

    if (comm := get_comm_world()) is not None:
        step = np.max(comm.allgather(step))
        redshift = max(comm.allgather(redshift))
        z_ranges = comm.allgather(redshift_range)
        z_min = min(zr[0] for zr in z_ranges)
        z_max = max(zr[1] for zr in z_ranges)
        redshift_range = (z_min, z_max)

    # lightcones are identified by their upper redshift slice
    header_schema = datasets[0].header.dump()
    header_schema.attributes["file"]["redshift"] = redshift
    header_schema.attributes["file"]["step"] = step
    header_schema.attributes["lightcone"]["z_range"] = redshift_range
    return header_schema


def stack_lightcone_datasets_in_schema(
    datasets: list[ds.Dataset], name: str, redshift_range: tuple[float, float]
):
    if len(datasets) == 1 and get_comm_world() is None:
        schema = datasets[0].make_schema(name=name)
        header = sync_headers(datasets, redshift_range)
        schema.children["header"] = header
        return schema

    schemas = [ds.make_schema(name=name) for ds in datasets]
    index_names = list(schemas[0].children["index"].children.keys())
    index_names.sort()
    max_level = int(index_names[-1][-1])

    assert all(isinstance(dataset, ds.Dataset) for dataset in datasets)
    new_data_group = stack_data_groups([schema.children["data"] for schema in schemas])

    order = get_stacked_lightcone_order(datasets, max_level)
    updater = partial(update_order, order=order)

    for column in new_data_group.columns.values():
        column.set_transformation(updater)

    new_index_group = stack_index_groups(
        [schema.children["index"] for schema in schemas]
    )
    header_schema = sync_headers(datasets, redshift_range)

    children = {
        "data": new_data_group,
        "index": new_index_group,
        "header": header_schema,
    }

    return make_schema(name, FileEntry.DATASET, children=children)


def stack_index_groups(schemas: list[Schema]):
    base_schema = schemas[0]
    new_children = {}
    for index_level in base_schema.children.keys():
        all_level_schemas = [schema.children[index_level] for schema in schemas]
        new_children[index_level] = stack_data_groups(all_level_schemas)
    return make_schema("index", FileEntry.COLUMNS, new_children)


def stack_data_groups(schemas: list[Schema]):
    if len(schemas) == 1:
        return schemas[0]
    base_schema = schemas[0]
    new_writers = {}
    for name, column_writer in base_schema.columns.items():
        other_writers = [schema.columns[name] for schema in schemas[1:]]
        new_writer = column_writer.combine(other_writers)
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
    coordinates = list(filter(lambda coord_list: len(coord_list) > 0, coordinates))
    

    pixels = np.concatenate(
        [
            hp.ang2pix(nside, coords.ra.value, coords.dec.value, lonlat=True, nest=True)
            for coords in coordinates
        ]
    )
    return np.argsort(pixels)
