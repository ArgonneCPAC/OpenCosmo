from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Optional

import h5py
import numpy as np
from mpi4py import MPI

from opencosmo.io.schema import FileEntry, Schema, make_schema
from opencosmo.io.verify import ZeroLengthError, verify_file
from opencosmo.io.writer import ColumnCombineStrategy
from opencosmo.mpi import get_comm_world

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.io.schema import Schema
    from opencosmo.io.writer import ColumnWriter



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
    """
    Main entry point for writing data in parallel. Proceeds
    in three steps:

    1. Verify the file schemas are valid separately
    2. Recursively verify the schemas can be made consistent across ranks
    3. Recursively write the data.
    """
    comm = get_comm_world()
    if comm is None:
        raise ValueError("Got a null comm!")
    paths = set(comm.allgather(file))
    if len(paths) != 1:
        raise ValueError("Different ranks recieved a different path to output to!")

    try:
        verify_file(file_schema)  # Initial verification
        results = comm.allgather(CombineState.VALID)
    except ValueError:
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

    verify_schemas(file_schema, new_comm)
    try:
        with h5py.File(file, "w", driver="mpio", comm=new_comm) as f:
            __write(file_schema, f, new_comm)

    except ValueError:  # parallell hdf5 not available
        raise NotImplementedError(
            "MPI writes without paralell hdf5 are not yet supported"
        )
    cleanup_mpi(comm, new_comm, new_group)


def cleanup_mpi(comm_world: MPI.Comm, comm_write: MPI.Comm, group_write: MPI.Group):
    comm_world.Barrier()
    if comm_write != MPI.COMM_NULL:
        comm_write.Free()
    group_write.Free()


def get_all_keys(data: dict, comm: MPI.Comm, debug=False):
    """
    Return all keys in the dictionary across all ranks, sorted
    alphabetically.
    """
    data_names = set(data.keys())
    all_data_names: Iterable[str]
    all_data_names = data_names.union(*comm.allgather(data_names))
    all_data_names = list(all_data_names)
    all_data_names.sort()
    return all_data_names


def verify_schemas(schema: Schema, comm: MPI.Comm) -> None:
    """
    By this stage, we know that all the ranks that are participating have a valid
    file schema. We now need to verify that they can be made consistent across ranks.

    This is done recursively. For each schema, we verify its columns and its attributes.
    Then we proceed to its children. Ranks that do not have data for a given child
    are simply excluded from the verification process.

    """

    if comm.Get_size() == 1:  # this shouldn't happen, but include anyway
        return

    file_types = set(comm.allgather(schema.type))
    if len(file_types) > 1:
        raise ValueError(
            "Unable to combine file schemas, as they do not have the same type!"
        )

    verify_columns(schema.columns, comm)
    verify_attributes(schema.attributes, comm)
    all_child_names = get_all_keys(schema.children if schema is not None else {}, comm)

    for child_name in all_child_names:
        has_child = comm.allgather(child_name in schema.children)
        if all(has_child):
            new_comm = comm
        else:
            ranks_to_include = [i for i in range(len(has_child)) if has_child[i]]
            group = comm.Get_group()
            new_group = group.Incl(ranks_to_include)
            new_comm = comm.Create(new_group)
        if child_name in schema.children:
            verify_schemas(schema.children[child_name], new_comm)


def verify_columns(columns: dict[str, ColumnWriter], comm: MPI.Comm):
    """
    Verify a group of columns in a schema. Note, the single-threaded verification
    process already ensures that all columns in the data group have the same
    length in any given rank, which guarantees this will also be the case when
    we combine columns across ranks. All we have to check here is:

    1. Are the shapes consistent
    2. Are the data types consistent
    3. Are the column attributes consistent

    """
    all_column_names = get_all_keys(columns, comm)
    for colname in all_column_names:
        if colname not in columns:
            # Simply ignore a rank that doesn't have this column
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
    all_metadata = comm.allgather(metadata)
    if not all(md == all_metadata[0] for md in all_metadata[1:]):
        raise ValueError("Not all ranks recieved the same metadata!")


def __write(schema: Schema, group: h5py.File | h5py.Group, comm: MPI.Comm):
    __write_columns(schema, group, comm)
    __write_metadata(schema, group, comm)
    all_child_names = get_all_keys(schema.children, comm)
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
    reference_meta = all_meta[0]

    if strategy == ColumnCombineStrategy.CONCAT:
        total_length = sum(m[0][0] for m in all_meta)
    else:
        total_length = reference_meta[0][0]
    shape = (total_length,) + reference_meta[0][1:]
    return shape, all_meta[0][1], all_meta[0][2]


def get_column_offset(column: Optional[ColumnWriter], comm: MPI.Comm):
    length = 0 if column is None else len(column)
    all_lengths = comm.allgather(length)
    offsets = np.insert(np.cumsum(all_lengths), 0, 0)
    return offsets[comm.Get_rank()]


def __write_metadata(schema: Schema, group: h5py.File | h5py.Group, comm: MPI.Comm):
    if schema.type == FileEntry.EMPTY:
        attrs = comm.allgather(None)
    else:
        attrs = comm.allgather(schema.attributes)
    attrs_to_write = list(filter(lambda at: at is not None, attrs))[0]
    for path, metadata in attrs_to_write.items():
        metadata_group = group.require_group(path)
        metadata_group.attrs.update(metadata)


def __write_columns(schema: Schema, group: h5py.File | h5py.Group, comm: MPI.Comm):
    all_column_names = get_all_keys(schema.columns, comm)
    for cn in all_column_names:
        writer = schema.columns.get(cn)
        shape, dtype, attrs = get_column_allocation_metadata(writer, comm)
        ds = group.create_dataset(cn, shape=shape, dtype=dtype)
        ds.attrs.update(attrs)
        __write_column(writer, ds, comm)


def __write_column(writer: Optional[ColumnWriter], ds: h5py.Dataset, comm: MPI.Comm):
    offset = get_column_offset(writer, comm)
    strategy = None if writer is None else writer.combine_strategy
    strategies = list(filter(lambda s: s is not None, comm.allgather(strategy)))

    match strategies[0]:
        case ColumnCombineStrategy.CONCAT:
            if writer is not None:
                data = writer.get_data(comm)
                ds.write_direct(data, dest_sel=np.s_[offset : offset + len(data)])
        case ColumnCombineStrategy.SUM:
            if writer is None:
                data = np.zeros(ds.shape, ds.dtype)
            else:
                data = writer.get_data(comm)
            data_to_write = comm.reduce(data)
            if comm.Get_rank() == 0:
                ds[:] = data_to_write

    comm.Barrier()
    ds.file.flush()
