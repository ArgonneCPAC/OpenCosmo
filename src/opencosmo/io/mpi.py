from __future__ import annotations

from collections import defaultdict
from copy import copy
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Type, TypeVar, cast

import h5py
import numpy as np
from mpi4py import MPI

from opencosmo.index import ChunkedIndex
from opencosmo.mpi import get_comm_world

from .schemas import (
    ColumnSchema,
    DatasetSchema,
    EmptyColumnSchema,
    FileSchema,
    LightconeSchema,
    SimCollectionSchema,
    StructCollectionSchema,
    ZeroLengthError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.header import OpenCosmoHeader

    from .protocols import DataSchema

"""
When working with MPI, datasets are chunked across ranks. Here we combine the schemas
from several ranks into a single schema that can be allocated by rank 0. Each 
rank will then write its own data to the specific section of the file 
it is responsible for.

As with schemas and writers, everything is very hierarcical here. A function
does some consistency checks, then calls a function that combines its children.

Ranks with different schemas are supported. For example, one rank may have data for one
dataset in a collection but not another. So long as the top-level structure is the same,
things will be handled.
"""


class CombineState(Enum):
    VALID = 1
    ZERO_LENGTH = 2
    INVALID = 3


def write_parallel(file: Path, file_schema: FileSchema):
    comm = get_comm_world()
    if comm is None:
        raise ValueError("Got a null comm!")
    paths = set(comm.allgather(file))
    if len(paths) != 1:
        raise ValueError("Different ranks recieved a different path to output to!")

    try:
        file_schema.verify()
        results = comm.allgather(CombineState.VALID)
    except ValueError as e:
        results = comm.allgather(CombineState.INVALID)
    except ZeroLengthError:
        results = comm.allgather(CombineState.ZERO_LENGTH)
    if not all(results):
        raise ValueError("One or more ranks recieved invalid schemas!")

    has_data = [i for i, state in enumerate(results) if state == CombineState.VALID]
    group = comm.Get_group()
    new_group = group.Incl(has_data)
    new_comm = comm.Create(new_group)
    if new_comm == MPI.COMM_NULL:
        return cleanup_mpi(comm, new_comm, new_group)
    rank = new_comm.Get_rank()

    new_schema = combine_file_schemas(file_schema, new_comm)
    if rank == 0:
        with h5py.File(file, "w") as f:
            new_schema.allocate(f)

    writer = new_schema.into_writer(new_comm)

    try:
        with h5py.File(file, "a", driver="mpio", comm=new_comm) as f:
            writer.write(f)

    except ValueError:  # parallell hdf5 not available
        raise NotImplementedError(
            "MPI writes without paralell hdf5 are not yet supported"
        )
        nranks = new_comm.Get_size()
        rank = new_comm.Get_rank()
        for i in range(nranks):
            if i == rank:
                with h5py.File(file, "a") as f:
                    writer.write(f)
            new_comm.Barrier()
    cleanup_mpi(comm, new_comm, new_group)


def cleanup_mpi(comm_world: MPI.Comm, comm_write: MPI.Comm, group_write: MPI.Group):
    comm_world.Barrier()
    if comm_write != MPI.COMM_NULL:
        comm_write.Free()
    group_write.Free()


def get_all_child_names(children: dict, comm: MPI.Comm, debug=False):
    child_names = set(children.keys())
    all_child_names: Iterable[str]
    all_child_names = child_names.union(*comm.allgather(child_names))
    all_child_names = list(all_child_names)
    all_child_names.sort()
    return all_child_names


def verify_structure(schemas: Mapping[str, DataSchema], comm: MPI.Comm):
    verify_names(schemas, comm)
    verify_types(schemas, comm)


def verify_names(schemas: Mapping[str, DataSchema], comm: MPI.Comm):
    names = set(schemas.keys())
    all_names = comm.allgather(names)
    if not all(ns == all_names[0] for ns in all_names[1:]):
        raise ValueError(
            "Tried to combine a collection of schemas with different names!"
        )


def verify_types(schemas: Mapping[str, DataSchema], comm: MPI.Comm):
    types = list(str(type(c)) for c in schemas.values())
    types.sort()
    all_types = comm.allgather(types)
    if not all(ts == all_types[0] for ts in all_types[1:]):
        raise ValueError(
            "Tried to combine a collection of schemas with different types!"
        )


def combine_file_schemas(schema: FileSchema, comm: MPI.Comm) -> FileSchema:
    if comm.Get_size() == 1:
        return schema

    all_child_names = get_all_child_names(
        schema.children if schema is not None else {}, comm
    )
    new_schema = FileSchema()

    for child_name in all_child_names:
        child = schema.children.get(child_name) if schema is not None else None
        new_child = combine_file_child(child, comm)
        new_schema.add_child(new_child, child_name)

    return new_schema


S = TypeVar("S", DatasetSchema, SimCollectionSchema, StructCollectionSchema)


def combine_file_child(schema: S | None, comm: MPI.Comm) -> S:
    match schema:
        case DatasetSchema():
            return cast("S", combine_dataset_schemas(schema, comm))
        case SimCollectionSchema():
            return cast("S", combine_simcollection_schema(schema, comm))
        case StructCollectionSchema():
            return cast("S", combine_structcollection_schema(schema, comm))
        case LightconeSchema():
            return cast("S", combine_lightcone_schema(schema, comm))
        case _:
            raise ValueError(f"Invalid file child of type {type(schema)}")


def validate_headers(
    header: OpenCosmoHeader | None, comm: MPI.Comm, header_updates: dict = {}
):
    all_headers: Iterable[OpenCosmoHeader] = comm.allgather(header)
    all_headers = filter(lambda h: h is not None, all_headers)
    all_headers = list(map(lambda h: h.with_parameters(header_updates), all_headers))
    regions = set([h.file.region for h in all_headers])
    if len(regions) > 1:
        all_headers = [h.with_region(None) for h in all_headers]

    if any(h != all_headers[0] for h in all_headers[1:]):
        raise ValueError("Not all datasets have the same header!")
    return all_headers[0]


def combine_dataset_schemas(
    schema: DatasetSchema | None,
    comm: MPI.Comm,
    header_updates: dict = {},
) -> DatasetSchema:
    if schema is not None:
        header = validate_headers(schema.header, comm, header_updates)
        columns = schema.columns
    else:
        header = validate_headers(None, comm, header_updates)
        columns = defaultdict(dict)

    children = schema.columns if schema is not None else {}
    new_schema = DatasetSchema(header=header)
    all_group_names = get_all_child_names(children, comm)

    for groupname in all_group_names:
        group_column_names = get_all_child_names(children.get(groupname, {}), comm)
        if groupname in ["data", "data_linked"]:
            new_schema.columns[groupname] = combine_data_group(
                columns[groupname], group_column_names, comm
            )
        elif groupname == "index":
            new_schema.columns[groupname] = combine_spatial_index_schema(
                columns[groupname], comm
            )
    return new_schema


def combine_data_group(columns: dict, order: list[str], comm: MPI.Comm):
    output = {}
    for colname in order:
        column = columns.get(colname)
        assert not isinstance(column, EmptyColumnSchema)
        new_column_schema = combine_column_schemas(column, comm)

        output[colname] = new_column_schema
    return output


def combine_spatial_index_schema(
    columns: dict[str, ColumnSchema], comm: MPI.Comm = MPI.COMM_WORLD
):
    has_schema = len(columns) > 0
    all_has_schema = comm.allgather(has_schema)

    if not any(all_has_schema):
        return None

    levels = set(map(lambda key: int(key.split("/")[0][-1]), columns.keys()))
    if not levels:
        n_levels = -1
    else:
        n_levels = max(levels)
    all_max_levels = set(comm.allgather(n_levels))
    if -1 in all_max_levels:
        all_max_levels.remove(-1)

    if len(set(all_max_levels)) != 1:
        raise ValueError("Schemas for all ranks must have the same number of levels!")

    max_level = all_max_levels.pop()
    output = {}
    for level in range(max_level + 1):
        level_schemas = {
            key: val for key, val in columns.items() if f"level_{level}" in key
        }
        new_level_schemas = combine_spatial_index_level_schemas(
            level_schemas, level, comm
        )
        output.update(new_level_schemas)

    return output


def combine_spatial_index_level_schemas(
    schemas: dict[str, ColumnSchema], level: int, comm: MPI.Comm
):
    if schemas:
        assert len(schemas) == 2
        start = schemas[f"level_{level}/start"]
        size = schemas[f"level_{level}/size"]
        start_len = len(start)
        size_len = len(size)
    else:
        start_len = 0
        size_len = 0

    all_start_lens = set(filter(lambda s: s is not None, comm.allgather(start_len)))
    all_size_lens = set(filter(lambda s: s is not None, comm.allgather(size_len)))
    if 0 in all_start_lens:
        all_start_lens.remove(0)
        all_size_lens.remove(0)

    if all_start_lens != all_size_lens or len(all_start_lens) != 1:
        raise ValueError("Invalid starts and sizes")

    level_len = all_start_lens.pop()

    if not schemas:
        source = np.zeros(level_len, dtype=np.int32)
        index = ChunkedIndex.from_size(len(source))
        start = ColumnSchema(
            f"level_{level}/start", index, source, {}, total_length=level_len
        )
        size = ColumnSchema(
            f"level_{level}/size", index, source, {}, total_length=level_len
        )

    else:
        start = ColumnSchema(
            f"level_{level}/start",
            start.index,
            start.source,
            {},
            total_length=level_len,
        )
        size = ColumnSchema(
            f"level_{level}/size",
            size.index,
            size.source,
            {},
            total_length=level_len,
        )

    new_schemas = {f"level_{level}/start": start, f"level_{level}/size": size}
    return new_schemas


def combine_lightcone_schema(schema: LightconeSchema | None, comm: MPI.Comm):
    if schema is None:
        children = {}
    else:
        children = schema.children

    all_child_names = get_all_child_names(children, comm, debug=True)
    new_schema = LightconeSchema()

    for child_name in all_child_names:
        child = children.get(child_name)
        z_range = get_z_range(child, comm)

        new_dataset_schema = combine_dataset_schemas(
            children.get(child_name), comm, {"lightcone/z_range": z_range}
        )
        new_schema.add_child(new_dataset_schema, child_name)
    return new_schema


def get_z_range(ds: DatasetSchema | None, comm: MPI.Comm):
    if ds is not None and ds.header is not None:
        z_ranges = comm.allgather(ds.header.lightcone["z_range"])
    else:
        z_ranges = comm.allgather(None)
    z_ranges = list(filter(lambda dz: dz is not None, z_ranges))
    dzs: Iterable[float] = map(lambda dz: dz[1] - dz[0], z_ranges)
    dzs = list(dzs)
    max_idx = np.argmax(dzs)
    return list(z_ranges)[max_idx]


def combine_simcollection_schema(
    schema: SimCollectionSchema | None, comm: MPI.Comm
) -> SimCollectionSchema:
    if schema is None:
        children = {}
    else:
        children = schema.children
    all_child_names = get_all_child_names(children, comm)

    new_schema = SimCollectionSchema()
    new_child: DatasetSchema | StructCollectionSchema

    for child_name in all_child_names:
        child = children.get(child_name)
        new_child = combine_dataset_schemas(child, comm)
        new_schema.add_child(new_child, child_name)
    return new_schema


def combine_structcollection_schema(
    schema: StructCollectionSchema, comm: MPI.Comm
) -> StructCollectionSchema:
    child_names: Iterable[str] = set(schema.children.keys())
    all_child_names = comm.allgather(child_names)
    if not all(cns == all_child_names[0] for cns in all_child_names[1:]):
        raise ValueError(
            "Tried to combine ismulation collections with different children!"
        )

    child_types = set(str(type(c)) for c in schema.children.values())
    all_child_types = comm.allgather(child_types)
    if not all(cts == all_child_types[0] for cts in all_child_types[1:]):
        raise ValueError(
            "Tried to combine ismulation collections with different children!"
        )

    new_schema = StructCollectionSchema()
    child_names = list(child_names)
    child_names.sort()
    new_child: DatasetSchema | StructCollectionSchema

    for i, name in enumerate(child_names):
        cn = comm.bcast(name)
        child = schema.children[cn]
        if isinstance(child, DatasetSchema):
            new_child = combine_dataset_schemas(child, comm)
        elif isinstance(child, StructCollectionSchema):
            new_child = combine_structcollection_schema(child, comm)
        else:
            raise ValueError(
                "Found a child of a structure collection that was not a Dataset!"
            )
        new_schema.add_child(new_child, cn)

    return new_schema


def verify_column_schemas(schema: ColumnSchema | None, comm: MPI.Comm):
    if schema is None:
        data = comm.allgather(None)
    else:
        data = comm.allgather(
            (schema.source.shape, schema.name, dict(schema.attrs), schema.source.dtype)
        )

    data = list(filter(lambda elem: elem is not None, data))
    if any(d[1:] != data[0][1:] for d in data[1:]):
        raise ValueError("Tried to write incompatible columns to the same output!")
    return data[0]


def combine_column_schemas(
    schema: ColumnSchema | None, comm: MPI.Comm
) -> ColumnSchema | EmptyColumnSchema:
    rank = comm.Get_rank()
    if schema is None:
        length = 0
    else:
        length = len(schema.index)

    shape, name, attrs, dtype = verify_column_schemas(schema, comm)

    lengths = comm.allgather(length)
    total_length = np.sum(lengths)
    rank_offsets = np.insert(np.cumsum(lengths), 0, 0)[:-1]
    rank_offset = rank_offsets[rank]

    new_schema: ColumnSchema | EmptyColumnSchema
    if schema is None:
        new_schema = EmptyColumnSchema(name, attrs, dtype, (total_length,) + shape[1:])
    else:
        new_schema = ColumnSchema(
            schema.name,
            schema.index,
            schema.source,
            schema.attrs,
            total_length=total_length,
        )
        if length != 0:
            new_schema.set_offset(rank_offset)
    return new_schema
