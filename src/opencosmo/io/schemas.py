from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Any, Iterable, Optional

import h5py
import hdf5plugin  # type: ignore

import opencosmo.io.writers as iow
from opencosmo.index import ChunkedIndex

if TYPE_CHECKING:
    import numpy as np
    from mpi4py import MPI
    from numpy.typing import DTypeLike, NDArray

    import opencosmo.io.protocols as iop
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex

ColumnShape = tuple[int, ...]

"""
Schemas represent the hierarchy of groups and datasets in a given hdf file.

Schemas may have child schemas, and must be able to allocate themselves
in an open HDF5 file. Allocation may look as simple as creating a particular
group and passing that group to child allocators. In the case of a ColumnSchema,
actual creation of hdf5 datasets is performed, complete with its length, dtype
and compressin.

Any validaty tests across multiple children must be performed by the parent schema.
For example, an OpenCosmoDataset must have columns with compatible shapes. It is the
responsibility of the DatasetSchema to enforce this.


Note that "Dataset" is used in the opencosmo sense here so:

hdf5.Dataset == opencosmo Column
hdf5.Group == opencosmo Dataset (sometimes)

Schemas also must define an into_writer method that creates the object that will
actually put data into the file. As a result, certain schemas must hold references
to the underlying data they reference.

See opencosmo.io.protocols for required methods.
"""

COMPRESSION = hdf5plugin.Blosc2()


class FileSchema:
    """
    Always the top level of schema. Has very few responsibilites
    besides holding children.
    """

    def __init__(self):
        self.children = {}

    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.children:
            raise ValueError(f"File schema already has child named {name}")
        match child:
            case (
                SimCollectionSchema()
                | StructCollectionSchema()
                | DatasetSchema()
                | LightconeSchema()
                | EmptyColumnSchema()
            ):
                self.children[name] = child
            case _:
                raise ValueError(
                    f"File schema cannot have children of type {type(child)}"
                )

    def allocate(self, group: h5py.File | h5py.Group):
        self.verify()
        if not isinstance(group, h5py.File):
            raise ValueError(
                "File Schema allocation must be done at the top level of a h5py file!"
            )
        if len(self.children) == 1:
            ds = next(iter(self.children.values()))
            return ds.allocate(group)

        for ds_name, ds in self.children.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)

    def verify(self):
        if len(self.children) == 0:
            raise ValueError("This file schema has no children!")
        for child in self.children.values():
            child.verify()

    def into_writer(self, comm: Optional["MPI.Comm"] = None):
        children = {
            name: schema.into_writer(comm) for name, schema in self.children.items()
        }
        return iow.FileWriter(children)


class SimCollectionSchema:
    """
    A schema for simulation collections. Like FileSchema, it has very few
    responsibilites outside of holding children and making sure they are
    valid types.
    """

    def __init__(self):
        self.children = {}

    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.children:
            raise ValueError(f"SimCollection already has a dataset with name {name}")
        match child:
            case StructCollectionSchema() | DatasetSchema():
                self.children[name] = child
            case _:
                raise ValueError(
                    f"Sim collection cannot take children of type {type(child)}"
                )

    def allocate(self, group: h5py.File | h5py.Group):
        if not isinstance(group, h5py.File):
            raise ValueError(
                "File Schema allocation must be done at the top level of a h5py file!"
            )
        for ds_name, ds in self.children.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)

    def verify(self):
        zero_length = set()
        for name, child in self.children.items():
            try:
                child.verify()
            except ZeroLengthError:
                zero_length.add(name)
        if len(zero_length) == len(self.children):
            raise ZeroLengthError

    def into_writer(self, comm: Optional["MPI.Comm"] = None):
        children = {
            name: child.into_writer(comm) for name, child in self.children.items()
        }
        return iow.CollectionWriter(children)


class LightconeSchema:
    def __init__(self):
        self.children: dict[str, DatasetSchema] = {}

    def verify(self):
        zero_length = set()
        for name, child in self.children.items():
            try:
                child.verify()
            except ZeroLengthError:
                zero_length.add(name)
        if len(zero_length) == len(self.children):
            raise ZeroLengthError

    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.children:
            raise ValueError(f"LightconeSchema already has child with name {name}")
        match child:
            case DatasetSchema():
                self.children[name] = child
            case _:
                raise ValueError(
                    f"LightconeSchema cannot take children of type {type(child)}"
                )

    def allocate(self, group: h5py.Group):
        if len(self.children) == 1:
            next(iter(self.children.values())).allocate(group)
            return
        for ds_name, ds in self.children.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)

    def into_writer(self, comm: Optional["MPI.Comm"] = None):
        dataset_writers = {
            key: val.into_writer(comm) for key, val in self.children.items()
        }
        return iow.CollectionWriter(dataset_writers)


class StructCollectionSchema:
    """
    Schema for structure collections. Actual linking data is held by datasets,
    so the main responsibility of this schema is to ensure that at least one
    link exists across all its datasets.

    Structure collections always have a header, because they always
    contain several datasets from the same simulation.
    """

    def __init__(self):
        self.children: dict[str, DatasetSchema | StructCollectionSchema] = defaultdict(
            dict
        )

    def verify(self):
        if len(self.children) < 2:
            raise ValueError("StructCollection must have at least two children!")

        found_links = False
        zero_length = set()
        for name, child in self.children.items():
            try:
                child.verify()
            except ZeroLengthError:
                zero_length.add(name)
                continue
            linked_columns = filter(lambda col: "data_linked" in col, child.children)
            if list(linked_columns):
                found_links = True

        if not found_links:
            raise ValueError("StructCollection must get at least one link!")
        elif len(zero_length) == len(self.children):
            raise ZeroLengthError

    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.children:
            raise ValueError(
                f"StructCollectionSchema already has child with name {name}"
            )
        match child:
            case DatasetSchema():
                self.children[name] = child
            case StructCollectionSchema():
                for key, grandchild in child.children.items():
                    if key in self.children:
                        self.children[f"{name}_{key}"] = grandchild
                    else:
                        self.children[key] = grandchild
            case _:
                raise ValueError(
                    f"StructCollectionSchema cannot take children of type {type(child)}"
                )

    def allocate(self, group: h5py.Group):
        if len(self.children) == 1:
            raise ValueError("A dataset collection cannot have only one member!")
        for ds_name, ds in self.children.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)

    def into_writer(self, comm: Optional["MPI.Comm"] = None):
        dataset_writers = {
            key: val.into_writer(comm) for key, val in self.children.items()
        }
        return iow.CollectionWriter(dataset_writers)


class ZeroLengthError(Exception):
    pass


class DatasetSchema:
    """
    A schema for an opencosmo.dataset. This schema may or may not have a header.
    For example, a StructureCollection has a single collection for all datasets,
    so it does not need to be handled by the datasets it holds.
    """

    def __init__(
        self,
        header: Optional[OpenCosmoHeader] = None,
    ):
        self.columns: dict[str, dict[str, ColumnSchema]] = defaultdict(dict)
        self.header = header

    @property
    def children(self):
        return self.columns

    @classmethod
    def make_schema(
        cls,
        sources: dict[str, h5py.Group],
        columns: Iterable[str],
        index: DataIndex,
        header: Optional[OpenCosmoHeader] = None,
    ):
        schema = DatasetSchema(header)
        columns = set(columns)
        for colname in columns:
            colname_parts = colname.split("/", maxsplit=1)
            if (
                colname_parts[0] not in sources
                or colname_parts[1] not in sources[colname_parts[0]].keys()
            ):
                raise ValueError("Found a column with no source!")

            # Still need this to get metadata if the column is cached
            colsource = sources[colname_parts[0]][colname_parts[1]]

            column_schema = ColumnSchema(
                colname_parts[1], index, colsource, colsource.attrs
            )
            schema.add_child(column_schema, colname)
        return schema

    def add_child(self, child: iop.DataSchema, child_id: str):
        if not isinstance(child, ColumnSchema):
            raise ValueError("Dataset schemas can only take columns as childre")
        colparts = child_id.split("/", maxsplit=1)

        if colparts[1] in self.columns[colparts[0]]:
            raise ValueError(f"DatasetSchema already has a child named {child_id}")
        self.columns[colparts[0]][colparts[1]] = child

    def verify(self):
        if len(self.columns) == 0:
            raise ValueError("Datasets must have at least one column")

        column_lengths = set()
        for group, columns in self.columns.items():
            sizes = set(len(col) for col in columns.values())

        if len(sizes) == 1 and column_lengths.pop() == 0:
            raise ZeroLengthError()

        for group in self.columns.values():
            for column in group.values():
                column.verify()

    def allocate(self, group: h5py.File | h5py.Group):
        for groupname, columns in self.columns.items():
            data_group = group.require_group(groupname)
            for colname, column in columns.items():
                column.allocate(data_group)
        if self.header is not None:
            self.header.write(group)

    def into_writer(self, comm: Optional["MPI.Comm"] = None):
        colnames = list(self.columns.keys())
        colnames.sort()

        writers = {}
        for group, columns in self.columns.items():
            writers[group] = {
                name: cs.into_writer(comm) for name, cs in columns.items()
            }
        return iow.DatasetWriter(writers, comm)


class ColumnSchema:
    """
    This is where the magic actually happens. The ColumnSchema actually allocates
    space in the file holds a reference to the data that will eventually be written.
    It is also used eternally by the link schemas.
    """

    def __init__(
        self,
        name: str,
        index: DataIndex,
        source: h5py.Dataset | NDArray,
        attrs: dict[str, Any],
        total_length: Optional[int] = None,
    ):
        self.name = name
        self.index = index
        self.source = source
        self.attrs = attrs
        self.offset = 0
        if total_length is None:
            total_length = len(index)
        self.total_length = total_length

    def __len__(self):
        return len(self.index)

    def concatenate(self, *others: "ColumnSchema"):
        for other in others:
            if other.name != self.name:
                raise ValueError("Tried to combine columns with different names")
            if self.source.shape[1:] != other.source.shape[1:]:
                raise ValueError("Tried to combine columns with incompatible shapes")

        new_index = self.index.concatenate(*[o.index for o in others])
        return ColumnSchema(self.name, new_index, self.source, self.attrs)

    def add_child(self, *args, **kwargs):
        raise TypeError("Columns do not take children!")

    insert = add_child

    def set_offset(self, offset: int):
        self.offset = offset

    def verify(self):
        return True

    def allocate(self, group: h5py.Group):
        shape = (self.total_length,) + self.source.shape[1:]
        group.require_dataset(self.name, shape, self.source.dtype)
        for name, attr in self.attrs.items():
            group[self.name].attrs[name] = attr

    def into_writer(self, comm: Optional["MPI.Comm"] = None):
        return iow.ColumnWriter(self.name, self.index, self.source, self.offset)


class EmptyColumnSchema:
    """
    Represents a column that contains no data. The ONLY time this should be used
    is if we are writing in an MPI context and one or more of the ranks
    does not have data for a given column. However it must still know the column
    exists because structural work must be performed by all ranks in
    parallel HDF5.

    Its associated writer is effectively a no-op.

    """

    def __init__(
        self,
        name: str,
        attrs: dict[str, Any],
        dtype: DTypeLike,
        shape: tuple[int, ...],
    ):
        self.name = name
        self.attrs = attrs
        self.dtype = dtype
        self.shape = shape

    def __len__(self):
        return self.shape[0]

    def add_child(self, *args, **kwargs):
        raise TypeError("Columns do not take children!")

    insert = add_child

    def set_offset(self, offset: int):
        return

    def verify(self):
        return True

    def allocate(self, group: h5py.Group):
        group.create_dataset(self.name, self.shape, self.dtype)
        for name, attr in self.attrs.items():
            group[self.name].attrs[name] = attr

    def into_writer(self, comm: Optional["MPI.Comm"] = None):
        return iow.EmptyColumnWriter(self.name)
