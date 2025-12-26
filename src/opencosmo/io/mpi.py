from __future__ import annotations

from collections import defaultdict
from copy import copy
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Type, TypeVar, cast

import h5py
import numpy as np
from mpi4py import MPI

from opencosmo.index import from_size, get_length
from opencosmo.io.allocate import write_metadata
from opencosmo.io.schema import FileEntry, Schema, make_schema
from opencosmo.io.verify import verify_file
from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter
from opencosmo.mpi import get_comm_world

from .schemas import (
    ColumnSchema,
    DatasetSchema,
    EmptyColumnSchema,
    FileSchema,
    LightconeSchema,
    SimCollectionSchema,
    StackedLightconeDatasetSchema,
    StructCollectionSchema,
    ZeroLengthError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.schema import Schema

    from .protocols import DataSchema


"""
When working with MPI, datasets are chunked across ranks. Here we combine the schemas
from several ranks into a single schema that can be allocated by rank 0. Each 
rank will then write its own data to the specific section of the file 
it is responsible for.

When writing data with MPI, there are basically 3 things we have to verify in order to 
determine if everything is valid.

1. Is the top-level file structure the same for all ranks (e.g. lightcone? dataset?).
2. Do all columns that are going to be written to by two or more ranks have the same data type and compatible shapes?
3. Is metadata consistent across ranks? If not, are there rules in place to combine/update the fields?

If all three of these checks pass, it is guaranteed we can create a schema that can accomodate the data being written.

File schemas are simply collections of columns and metadata. Columns contain:

1. A reference to the underying data that will be written (either an h5py dataset or a numpy array)
2. An index which tells us which elements in 1 we are going to actually write
3. Possibly an output index, which tells us where in the output we are going to actually write to.
3. Possibly a function to update those values before writing. For example, a spatial index should be summed across ranks rather than concatenated.

In order to avoid MPI deadlocks, we always sort columns in alphabetical order before performing operations on the file.
"""


class CombineState(Enum):
    VALID = 1
    ZERO_LENGTH = 2
    INVALID = 3


def write_parallel(file: Path, file_schema: Schema):
    comm = get_comm_world()
    if comm is None:
        raise ValueError("Got a null comm!")
    paths = set(comm.allgather(file))
    if len(paths) != 1:
        raise ValueError("Different ranks recieved a different path to output to!")

    try:
        verify_file(file_schema)
        results = comm.allgather(CombineState.VALID)
    except ValueError as e:
        results = comm.allgather(CombineState.INVALID)
    except ZeroLengthError:
        results = comm.allgather(CombineState.ZERO_LENGTH)
    if any(rs == CombineState.INVALID for rs in results):
        raise ValueError("One or more ranks recieved invalid schemas!")

    has_data = [i for i, state in enumerate(results) if state == CombineState.VALID]
    if len(has_data) == 0:
        raise ValueError("No ranks have any data to write!")

    group = comm.Get_group()
    new_group = group.Incl(has_data)
    new_comm = comm.Create(new_group)
    if new_comm == MPI.COMM_NULL:
        return cleanup_mpi(comm, new_comm, new_group)
    rank = new_comm.Get_rank()

    verify_file_schemas(file_schema, new_comm)
    try:
        with h5py.File(file, "w", driver="mpio", comm=new_comm) as f:
            __write(file_schema, f, new_comm)

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


def get_all_child_names(children: dict, comm: MPI.Comm, debug=False):
    child_names = set(children.keys())
    all_child_names: Iterable[str]
    all_child_names = child_names.union(*comm.allgather(child_names))
    all_child_names = list(all_child_names)
    all_child_names.sort()
    return all_child_names


def verify_file_schemas(schema: Schema, comm: MPI.Comm) -> FileSchema:
    """
    By this stage, we know that all the ranks that are participating have a valid
    file schema. We now need to combine them so that:

    1. We know the combined shapes of all the columns, so we can allocate accordingly
    2. Each rank knows what column chunk it needs to write to

    """

    if comm.Get_size() == 1:  # this shouldn't happen, but include anyway
        return schema

    file_types = set(comm.allgather(schema.type))
    if len(file_types) > 1:
        raise ValueError(
            "Unable to combine file schemas, as they do not have the same type!"
        )

    verify_columns(schema.columns, comm)
    verify_attributes(schema.attributes, comm)
    all_child_names = get_all_child_names(
        schema.children if schema is not None else {}, comm
    )

    for child_name in all_child_names:
        has_child = comm.allgather(child_name in schema.children)
        if all(has_child):
            new_comm = comm
        else:
            ranks_to_include = [i for i in range(len(has_child)) if has_child[i]]
            group = comm.Get_group()
            new_group = group.Incl(has_data)
            new_comm = comm.Create(new_group)
        if child_name in schema.children:
            verify_file_schemas(schema.children[child_name], new_comm)


def verify_columns(columns: dict[str, ColumnWriter], comm: MPI.Comm):
    all_column_names = get_all_child_names(columns, comm)
    for colname in all_column_names:
        if colname not in columns:
            colmeta = comm.allgather(None)
        else:
            column = columns[colname]
            data_to_send = (
                column.combine_strategy,
                column.shape,
                column.dtype,
                column.attrs,
            )
            colmeta = comm.allgather(data_to_send)

        combine_strategies = set([cm[0] for cm in colmeta if cm is not None])
        if len(combine_strategies) > 1:
            raise ValueError("Combine strategy must be the same accross ranks!")

        shapes = set([cm[1][1:] for cm in colmeta if cm is not None])
        if len(shapes) > 1:
            raise ValueError(
                f"Column {colname} did not have consistent shapes across ranks!"
            )
        dtypes = set([cm[2] for cm in colmeta if cm is not None])
        if len(dtypes) > 1:
            raise ValueError(
                f"Column {colname} did not have consistent dtypes across ranks!"
            )
        attrs = [cm[3] for cm in colmeta if cm is not None]
        if any(attr_set != attrs[0] for attr_set in attrs[1:]):
            raise ValueError("Metadata was not consistent across ranks!")


def verify_attributes(metadata: dict[str, Any], comm: MPI.Comm):
    metadata = comm.allgather(metadata)
    if not all(md == metadata[0] for md in metadata[1:]):
        raise ValueError(f"Not all ranks recieved the same metadata!")


def __write(schema: Schema, group: h5py.File | h5py.Group, comm: MPI.Comm):
    all_column_names = get_all_child_names(schema.columns, comm)
    for cn in all_column_names:
        writer = schema.columns.get(cn)
        shape, dtype, attrs = get_column_allocation_metadata(writer, comm)
        ds = group.create_dataset(cn, shape=shape, dtype=dtype)
        ds.attrs.update(attrs)
        __write_column(writer, ds, comm)
    for path, metadata in schema.attributes.items():
        metadata_group = group.require_group(path)
        metadata_group.attrs.update(metadata)

    all_child_names = get_all_child_names(schema.children, comm)
    for cn in all_child_names:
        child_schema = schema.children.get(cn, make_schema(cn, FileEntry.EMPTY))
        new_group = group.require_group(cn)
        __write(child_schema, new_group, comm)


def get_column_allocation_metadata(column: Optional[ColumnWriter], comm: MPI.Comm):
    """
    This does NOT do any verification. It assumes you have already done that.
    """
    strategy = None if column is None else column.combine_strategy
    strategies = list(filter(lambda s: s is not None, comm.allgather(strategy)))
    strategy = strategies[0]

    meta = None if column is None else (column.shape, column.dtype, column.attrs)
    all_meta = list(filter(lambda cm: cm is not None, comm.allgather(meta)))
    if strategy == ColumnCombineStrategy.CONCAT:
        total_length = sum(meta[0][0] for meta in all_meta)
    else:
        total_length = meta[0][0]
    shape = (total_length,) + all_meta[0][0][1:]
    return shape, all_meta[0][1], all_meta[0][2]


def get_column_offset(column: Optiona[ColumnWriter], comm: MPI.Comm):
    length = 0 if column is None else len(column)
    all_lengths = comm.allgather(length)
    offsets = np.insert(np.cumsum(all_lengths), 0, 0)
    return offsets[comm.Get_rank()]


def __write_metadata(
    metadata: Optional[dict[str, Any]], group: h5py.File | h5py.Group, comm: MPI.Comm
):
    all_metadata = filter(lambda md: md is not None, comm.allgather(metadata))

    for path, metadata in schema.attributes.items():
        metadata_group = group.require_group(path)
        metadata_group.attrs.update(metadata)


def __write_column(writer: Optional[ColumnWriter], ds: h5py.Dataset, comm: MPI.Comm):
    offset = get_column_offset(writer, comm)
    strategy = None if writer is None else writer.combine_strategy
    strategies = list(filter(lambda s: s is not None, comm.allgather(strategy)))

    match strategies[0]:
        case ColumnCombineStrategy.CONCAT:
            if writer is None:
                return
            data = writer.data
            ds.write_direct(data, dest_sel=np.s_[offset : offset + len(data)])
        case ColumnCombineStrategy.SUM:
            if writer is None:
                data = np.zeros(ds.shape, ds.dtype)
            else:
                data = writer.data
            data_to_write = comm.reduce(data)
            if comm.Get_rank() == 0:
                ds[:] = data_to_write
