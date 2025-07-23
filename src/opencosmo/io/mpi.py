from copy import copy
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Optional, TypeVar, cast

import h5py
import numpy as np
from mpi4py import MPI

from opencosmo.mpi import get_comm_world

from .protocols import DataSchema
from .schemas import (
    ColumnSchema,
    DatasetSchema,
    FileSchema,
    IdxLinkSchema,
    LightconeSchema,
    LinkSchema,
    SimCollectionSchema,
    SpatialIndexSchema,
    StartSizeLinkSchema,
    StructCollectionSchema,
    ZeroLengthError,
)

"""
When working with MPI, datasets are chunked across ranks. Here we combine the schemas
from several ranks into a single schema that can be allocated by rank 0. Each 
rank will then write it's own data to the specific section of the file 
it is responsible for.

As with schemas and writers, everything is very hierarcical here. A function
does some consistency checks, then calls a function that combines its children.
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
    except ValueError:
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
        new_schema.verify()
        with h5py.File(file, "w") as f:
            new_schema.allocate(f)

    new_comm.Barrier()
    writer = file_schema.into_writer(new_comm)

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
    verify_structure(schema.children, comm)
    if comm.Get_size() == 1:
        return schema

    new_schema = FileSchema()

    for child_name in schema.children:
        child_name_ = comm.bcast(child_name)
        child = schema.children[child_name_]
        new_child = combine_file_child(child, comm)
        new_schema.add_child(new_child, child_name)

    return new_schema


S = TypeVar("S", DatasetSchema, SimCollectionSchema, StructCollectionSchema)


def combine_file_child(schema: S, comm: MPI.Comm) -> S:
    match schema:
        case DatasetSchema():
            return cast(S, combine_dataset_schemas(schema, comm))
        case SimCollectionSchema():
            return cast(S, combine_simcollection_schema(schema, comm))
        case StructCollectionSchema():
            return cast(S, combine_structcollection_schema(schema, comm))
        case LightconeSchema():
            return cast(S, combine_lightcone_schema(schema, comm))


def combine_dataset_schemas(schema: DatasetSchema, comm: MPI.Comm) -> DatasetSchema:
    verify_structure(schema.columns, comm)
    verify_structure(schema.links, comm)

    new_schema = DatasetSchema(schema.header)

    for colname in schema.columns.keys():
        colname_ = comm.bcast(colname)
        new_column = combine_column_schemas(schema.columns[colname_], comm)
        new_schema.add_child(new_column, colname)

    if len(schema.links) > 0:
        new_links = combine_links(schema.links, comm)
        for name, link in new_links.items():
            new_schema.add_child(link, name)

    new_spatial_idx_schema = combine_spatial_index_schema(schema.spatial_index, comm)
    if new_spatial_idx_schema is not None:
        new_schema.add_child(new_spatial_idx_schema, "index")

    return new_schema


def combine_spatial_index_schema(
    schema: Optional[SpatialIndexSchema], comm: MPI.Comm = MPI.COMM_WORLD
):
    has_schema = schema is not None
    all_has_schema = comm.allgather(has_schema)

    if not any(all_has_schema):
        return None

    elif not all(all_has_schema):
        raise ValueError("Some of the datasets have spatial indices and others don't!")

    schema = cast(SpatialIndexSchema, schema)

    n_levels = max(schema.levels)
    all_max_levels = comm.allgather(n_levels)
    if len(set(all_max_levels)) != 1:
        raise ValueError("Schemas for all ranks must have the same number of levels!")

    return schema


def combine_links(
    links: dict[str, LinkSchema], comm: MPI.Comm
) -> Mapping[str, LinkSchema]:
    new_links: dict[str, LinkSchema] = {}
    for name, link in links.items():
        if isinstance(link, StartSizeLinkSchema):
            new_links[name] = combine_start_size_link_schema(link, comm)
        else:
            new_links[name] = combine_idx_link_schema(link, comm)

    return new_links


def combine_idx_link_schema(schema: IdxLinkSchema, comm: MPI.Comm) -> IdxLinkSchema:
    column_schema = combine_column_schemas(schema.column, comm)
    new_schema = copy(schema)
    new_schema.column = column_schema
    return new_schema


def combine_start_size_link_schema(
    schema: StartSizeLinkSchema, comm: MPI.Comm
) -> StartSizeLinkSchema:
    start_column_schema = combine_column_schemas(schema.start, comm)
    size_column_schema = combine_column_schemas(schema.size, comm)
    new_schema = copy(schema)
    new_schema.start = start_column_schema
    new_schema.size = size_column_schema
    return new_schema


def combine_lightcone_schema(schema: LightconeSchema, comm: MPI.Comm):
    verify_structure(schema.children, comm)
    child_names = list(schema.children.keys())
    new_schema = LightconeSchema()
    child_names.sort()
    for child_name in child_names:
        new_dataset_schema = combine_dataset_schemas(schema.children[child_name], comm)
        new_schema.add_child(new_dataset_schema, child_name)
    return new_schema


def combine_simcollection_schema(
    schema: SimCollectionSchema, comm: MPI.Comm
) -> SimCollectionSchema:
    verify_structure(schema.children, comm)

    child_names = schema.children.keys()

    new_schema = SimCollectionSchema()
    new_child: DatasetSchema | StructCollectionSchema

    for child_name in child_names:
        child_name_ = comm.bcast(child_name)
        child = schema.children[child_name_]
        match child:
            case StructCollectionSchema():
                new_child = combine_structcollection_schema(child, comm)
            case DatasetSchema():
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

    new_schema = StructCollectionSchema(schema.header)
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


def combine_column_schemas(schema: ColumnSchema, comm: MPI.Comm) -> ColumnSchema:
    rank = comm.Get_rank()
    lengths = comm.allgather(len(schema.index))
    rank_offsets = np.insert(np.cumsum(lengths), 0, 0)[:-1]
    rank_offset = rank_offsets[rank]
    schema.set_offset(rank_offset)

    indices = comm.allgather(schema.index)
    new_index = indices[0].concatenate(*indices[1:])

    return ColumnSchema(schema.name, new_index, schema.source, schema.attrs)
