from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Iterable, Optional

import healpy as hp
import numpy as np

from opencosmo import dataset as ds
from opencosmo.io.mpi import get_all_keys
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.mpi import get_comm_world
from opencosmo.spatial.check import find_coordinates_2d

if TYPE_CHECKING:
    from opencosmo.io.schema import Schema
    from opencosmo.mpi import MPI


def update_order(data: np.ndarray, comm: Optional[MPI.Comm], order: np.ndarray):
    if comm is not None:
        return update_global_order_mpi(data, comm, order)

    return data[order]


def update_global_order_mpi(data, comm, order):
    needs_global_reordering = comm.allgather(np.any((order < 0) | (order > len(order))))
    if not np.any(needs_global_reordering):
        return data[order]

    ends = np.cumsum(comm.allgather(len(order)))
    starts = np.insert(ends, 0, 0)
    global_order = order + starts[comm.Get_rank()]
    all_data = comm.allgather(data)
    return np.concat(all_data)[global_order]


def sync_metadata(dataset_schemas: list[Schema]):
    additional_metadata = [schema.attributes for schema in dataset_schemas]
    if not any(additional_metadata):
        return {}
    if not all(am == additional_metadata[0] for am in additional_metadata[1:]):
        raise ValueError("Datasets don't have the same metadata!")

    return additional_metadata[0]


def sync_headers(datasets: list[ds.Dataset], redshift_range):
    if not datasets and (comm := get_comm_world()) is not None:
        comm.allgather(0)
        comm.allgather(-1)
        comm.allgather((1000, -1))
        return

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
    datasets: dict[str, list[ds.Dataset]],
    name: Optional[str],
    redshift_range: Optional[tuple[float, float]],
):
    n_datasets = sum(len(lst) for lst in datasets.values())
    if n_datasets == 1 and get_comm_world() is None:
        dataset_list = next(iter(datasets.values()))

        schema = dataset_list[0].make_schema(name=name)
        header = sync_headers(dataset_list, redshift_range)
        schema.children["header"] = header
        return {"data": schema}

    schema_children = {}
    ds_groups = get_all_keys(datasets, get_comm_world())
    for ds_group in ds_groups:
        ds_list = datasets.get(ds_group, [])
        ds_list = list(filter(lambda ds: len(ds) > 0, ds_list))
        if len(ds_list) == 0:
            get_stacked_lightcone_order([], -1)
            sync_headers(ds_list, None)
            continue
        schemas = [ds.make_schema(name=name) for ds in ds_list]
        index_names = list(schemas[0].children["index"].children.keys())
        index_names.sort()
        max_level = int(index_names[-1][-1])

        assert all(isinstance(dataset, ds.Dataset) for dataset in ds_list)
        new_data_group = stack_data_groups(
            [schema.children["data"] for schema in schemas]
        )

        order = get_stacked_lightcone_order(ds_list, max_level)
        updater = partial(update_order, order=order)

        for column in new_data_group.columns.values():
            column.set_transformation(updater)

        new_index_group = stack_index_groups(
            [schema.children["index"] for schema in schemas]
        )
        header_schema = sync_headers(ds_list, redshift_range)
        additional_metadata = sync_metadata(schemas)

        children = {
            "data": new_data_group,
            "index": new_index_group,
            "header": header_schema,
        }

        schema_name = ds_group if len(datasets) > 1 else name
        assert schema_name is not None
        schema_children[schema_name] = make_schema(
            schema_name,
            FileEntry.LIGHTCONE,
            children=children,
            attributes=additional_metadata,
        )

    return schema_children


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


def get_order_mpi(pixels, comm):
    pixel_order = np.argsort(pixels)
    if len(pixels) > 0:
        pixel_ranges = comm.allgather((pixels[pixel_order[0]], pixels[pixel_order[-1]]))

    else:
        pixel_ranges = comm.allgather(None)

    pixel_ranges = [pr for pr in pixel_ranges if pr is not None]
    for i in range(len(pixel_ranges) - 1):
        if pixel_ranges[i][1] > pixel_ranges[i + 1][0]:
            break
    else:
        return pixel_order

    all_pixels = np.concat(comm.allgather(pixels))
    new_order = np.argsort(all_pixels)
    bounds = np.cumsum(comm.allgather(len(pixels)))
    bounds = np.insert(bounds, 0, 0)
    rank = comm.Get_rank()
    return new_order[bounds[rank] : bounds[rank + 1]] - bounds[rank]


def get_stacked_lightcone_order(datasets: Iterable[ds.Dataset], max_index_depth: int):
    datasets = list(datasets)
    nside = 2**max_index_depth
    coordinates = list(map(find_coordinates_2d, datasets))
    coordinates = list(filter(lambda coord_list: len(coord_list) > 0, coordinates))

    if datasets:
        pixels = np.concatenate(
            [
                hp.ang2pix(
                    nside, coords.ra.value, coords.dec.value, lonlat=True, nest=True
                )
                for coords in coordinates
            ]
        )

    else:
        pixels = np.array([])

    if (comm := get_comm_world()) is not None:
        return get_order_mpi(pixels, comm)

    return np.argsort(pixels)
