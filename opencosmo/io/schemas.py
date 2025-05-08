from typing import Iterable, Type, Optional
from enum import StrEnum, auto
import opencosmo.io.protocols as iop
import opencosmo.io.writers  as iow
from opencosmo.dataset.index import DataIndex
from opencosmo.header import OpenCosmoHeader

import h5py
import numpy as np
from numpy.typing import DTypeLike

ColumnShape = tuple[int, ...]

"""
Schemas represent the hierarchy of groups and datasets in a given hdf file.

Schemas may have child schemas, and must be able to allocate themselves
in an open HDF5 file. Allocation may look as simple as creating a particular
group and passing that group to child allocators. In the case of a ColumnSchema,
actual of hdf5 datasets is performed.

Any validaty tests across multiple children must be performed by the parent schema.
For example, an OpenCosmoDataset must have columns with compatible shapes. It is the
responsibility of the DatasetSchema to enforce this.

Some schemas are defined elsewhere in the library.

Note that "Dataset" is used in the opencosmo sense here so:

hdf5.Dataset == opencosmo Column
hdf5.Group == opencosmo Dataset (sometimes)

Schemas are NOT responsible for handling metadata. Those are handled by writers.
"""

class FileSchemaChildren(StrEnum):
    DATASET = "DATASET"
    SIM_COLLECTION = "SIM_COLLECTION"
    STRUCT_COLLECTION = "STRUCT_COLLECTION"

    def into(self) -> Type[iop.DataSchema]:
        match self:
            case FileSchemaChildren.DATASET:
                return DatasetSchema
            case FileSchemaChildren.SIM_COLLECTION:
                return SimCollectionSchema
            case FileSchemaChildren.STRUCT_COLLECTION:
                return StructCollectionSchema
    
class SimCollectionSchema:
    pass

class StructCollectionSchema:
    pass


class FileSchema:
    CHILD_TYPES = FileSchemaChildren
    def __init__(self):
        self.children = {}

    def insert(self, child: iop.DataSchema, path: str, type_: str):
        try:
            child_name, remaining_path = path.split(".", maxsplit=1)
            if child_name not in self.children:
                raise ValueError(f"File schema has no child {child_name}")
            self.children[child_name].insert(child, remaining_path, type_)
        except ValueError:
            self.add_child(child, path, type_)



    def add_child(self, child: iop.DataSchema, name: str, type_: str):
        child_type = self.CHILD_TYPES(type_.upper()).into()
        if not isinstance(child, child_type):
            raise ValueError(f"type_ is {type_} but the child is of type {type(child)}")

        if name in self.children:
            raise ValueError(f"Writer already has a dataset with name {name}")
        self.children[name] = child

    def allocate(self, group: h5py.File | h5py.Group):
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

    def into_writer(self):
        children = {name: schema.into_writer() for name, schema in self.children.items()}
        return iow.FileWriter(children)

        

class DatasetSchemaChildren(StrEnum):
    COLUMN = "COLUMN"
    LINK = "LINK"

    def into(self):
        match self:
            case DatasetSchemaChildren.COLUMN:
                return ColumnSchema
            case DatasetSchemaChildren.LINK:
                return LinkSchema

class DatasetSchema:
    CHILD_TYPES = DatasetSchemaChildren
    
    def __init__(
        self,
        header: Optional[OpenCosmoHeader] = None,
    ):
        self.columns: list[ColumnSchema] = []
        self.links: list[LinkSchema] = []
        self.header = header

    @classmethod
    def make_schema(cls, source: h5py.Group | h5py.File,  columns: Iterable[str], index: DataIndex, header: Optional[OpenCosmoHeader] = None):
        schema = DatasetSchema(header)
        for colname in columns:
            if colname not in source.keys():
                raise ValueError("Dataset source is missing some columns!")
            column_schema = ColumnSchema(colname, index, source[colname])
            schema.add_child(column_schema, colname, "column")
        return schema

    def insert(self, child: iop.DataSchema, path: str, type_: str):
        if "." in path:
            raise ValueError("Datasets do not have grandchildren!")
        return self.add_child(child, path, type_)


    def add_child(self, child: iop.DataSchema, name: str, type_: str):
        child_type = self.CHILD_TYPES(type_.upper()).into()
        if not isinstance(child, child_type):
            raise ValueError(f"type_ is {type_} but the child is of type {type(child)}")
        if name in [c.name for c in self.columns]:
            raise ValueError(f"Dataset already has a child with name {name}")

        if child_type == LinkSchema:
            self.links.append(child)
        elif child_type == ColumnSchema:
            self.columns.append(child)

    def verify(self):
        if len(self.columns) == 0:
            raise ValueError("Datasets must have at least one column")

        column_lengths = set(c.shape[0] for c in self.columns)
        if len(column_lengths) > 1:
            raise ValueError("Datasets columns must be the same length!")


        for column in self.columns:
            column.verify()
        for link in self.links:
            link.verify()


    def allocate(self, group: h5py.File | h5py.Group):
        data_group = group.require_group("data")
        for column in self.columns:
            column.allocate(data_group)
        for link in self.links:
            link_group = group.require_group("data_linked")
            link.allocate(link_group)

    def into_writer(self):
        column_writers = [col.into_writer() for col in self.columns]
        return iow.DatasetWriter(column_writers, self.header)


class ColumnSchema:
    def __init__(self, name: str, index: DataIndex, source: h5py.Dataset):
        self.name = name
        self.index = index
        self.source = source

    def add_child(self, *args, **kwargs):
        raise TypeError("Columns do not take children!")
    insert = add_child

    def verify(self):
        if len(self.index) == 0:
            raise ValueError("Columns cannot have zero length!")
        if len(self.index) > len(self.source):
            raise ValueError("The index is longer than its source!")

    def allocate(self, group: h5py.Group):
        shape = (len(self.index),) + self.source.shape[1:]
        group.require_dataset(self.name, shape, self.source.dtype)

    def into_writer(self):
        return iow.ColumnWriter(self.name, self.index, self.source)

class LinkSchema:
    def __init__(self, name: str, n: int, has_sizes: bool = False):
        self.n = n
        self.has_sizes = has_sizes
        self.name = name

    def add_child(self, *args, **kwargs):
        raise TypeError("Links do not take children!")
    insert = add_child

    def verify(self):
        if self.n == 0:
            raise ValueError("Links cannot have zero length")


    def allocate(self, group: h5py.Group):
        if self.has_sizes:
            start_name = f"{self.name}_start"
            size_name = f"{self.name}_size"
            group.create_dataset(start_name, shape=(self.n,), dtype=np.uint64)
            group.create_dataset(size_name, shape=(self.n,), dtype=np.uint64)
        else:
            name = f"{self.name}_idx"
            group.create_dataset(name, shape=(self.n,), dtype=np.uint64)

    def into_writer(self, source: h5py.File | h5py.Group):
        if not isinstance(source, h5py.Group):
            raise ValueError("Expected a h5py group to write links to!")





