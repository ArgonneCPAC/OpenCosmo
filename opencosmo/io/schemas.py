from itertools import chain
from typing import Iterable, Optional
import opencosmo.io.protocols as iop
import opencosmo.io.writers  as iow
from opencosmo.dataset.index import DataIndex
from opencosmo.header import OpenCosmoHeader

import h5py
import hdf5plugin
import numpy as np

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

COMPRESSION = hdf5plugin.Blosc2()

class FileSchema:

    def __init__(self):
        self.children = {}

    def concatenate(self, *others: "FileSchema"):
        child_names = set(self.children.keys())
        for other in others:
            other_child_names = set(other.children.keys())
            if other_child_names != child_names:
                raise ValueError("File schemas are incompatible")

        new_schema = FileSchema()
        for name in child_names:
            new_child_schema = self.children[name].concatenate(*[o.children[name] for o in others])
            new_schema.add_child(name, new_child_schema)
        return new_schema


    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.children:
            raise ValueError(f"File schema already has child named {name}")
        match child:
            case SimCollectionSchema() | StructCollectionSchema() | DatasetSchema():
                self.children[name] = child
            case _:
                raise ValueError(f"File schema cannot have children of type {type(child)}")

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

    def into_writer(self):
        children = {name: schema.into_writer() for name, schema in self.children.items()}
        return iow.FileWriter(children)


class SimCollectionSchema:
    def __init__(self):
        self.children = {}

    def concatenate(self, *others: "SimCollectionSchema"):
        child_names = set(self.children.keys())
        for other in others:
            other_child_names = set(other.children.keys())
            if other_child_names != child_names:
                raise ValueError("File schemas are incompatible")
            if not isinstance(other, SimCollectionSchema):
                raise ValueError("Tried to combine children of different types!")

        new_schema = SimCollectionSchema()
        for name in child_names:
            new_child_schema = self.children[name].concatenate(*[o.children[name] for o in others])
            new_schema.add_child(name, new_child_schema)
        return new_schema


    def insert(self, child: iop.DataSchema, path: str, type_: str):
        try:
            child_name, remaining_path = path.split(".", maxsplit=1)
            if child_name not in self.children:
                raise ValueError(f"File schema has no child {child_name}")
            self.children[child_name].insert(child, remaining_path, type_)
        except ValueError:
            self.add_child(child, path)

    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.children:
            raise ValueError(f"SimCollection already has a dataset with name {name}")
        match child:
            case StructCollectionSchema() | DatasetSchema():
                self.children[name] = child
            case _:
                raise ValueError(f"Sim collection cannot take children of type {type(child)}")

    def allocate(self, group: h5py.File | h5py.Group):
        if not isinstance(group, h5py.File):
            raise ValueError(
                "File Schema allocation must be done at the top level of a h5py file!"
            )
        for ds_name, ds in self.children.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)

    def verify(self):
        if len(self.children) < 2:
            raise ValueError("A SimulationCollection must have at least two children!")
        for child in self.children.values():
            child.verify()

    def into_writer(self):
        children = {name: child.into_writer() for name, child in self.children.items()}
        return iow.CollectionWriter(children)



class StructCollectionSchema:
    def __init__(self, header: OpenCosmoHeader):
        self.children: dict[str, DatasetSchema] = {}
        self.header = header

    def concatenate(self, *others: "StructCollectionSchema"):
        child_names = set(self.children.keys())
        for other in others:
            other_child_names = set(other.children.keys())
            if other_child_names != child_names:
                raise ValueError("File schemas are incompatible")
            if not isinstance(other, StructCollectionSchema):
                raise ValueError("Tried to combine children of different types!")

            if other.header != self.header:
                raise ValueError("Tried to combine struct collections with different headers!")
            

        new_schema = StructCollectionSchema()
        for name in child_names:
            new_child_schema = self.children[name].concatenate(*[o.children[name] for o in others])
            new_schema.add_child(name, new_child_schema)
        return new_schema


    def insert(self, child: iop.DataSchema, path: str):
        try:
            child_name, remaining_path = path.split(".", maxsplit=1)
            if child_name not in self.children:
                raise ValueError(f"File schema has no child {child_name}")
            self.children[child_name].insert(child, remaining_path)
        except ValueError:
            self.add_child(child, path)

    def verify(self):
        if len(self.children) < 2:
            raise ValueError("StructCollection must have at least two children!")

        found_links = False
        for child in self.children.values():
            child.verify()
            if child.links:
                found_links = True
        if not found_links:
            raise ValueError("StructCollection must get at least one link!")
            

    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.children:
            raise ValueError(f"StructCollectionSchema already has child with name {name}")
        match child:
            case DatasetSchema():
                self.children[name] = child
            case IdxLinkSchema() | StartSizeLinkSchema():
                raise ValueError("LinkSchemas need to be added to a DatasetSchema directly. Perhaps you meant to call insert?")
            case _: 
                raise ValueError(f"StructCollectionSchema cannot take children of type {type(child)}")


    def allocate(self, group: h5py.Group):
        if len(self.children) == 1:
            raise ValueError("A dataset collection cannot have only one member!")
        for ds_name, ds in self.children.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)

    def into_writer(self):
        dataset_writers = {key: val.into_writer() for key, val in self.children.items()}
        return iow.CollectionWriter(dataset_writers, self.header)

class DatasetSchema:
    def __init__(
        self,
        header: Optional[OpenCosmoHeader] = None,
    ):
        self.columns: dict[str, ColumnSchema] = {}
        self.links: dict[str, IdxLinkSchema | StartSizeLinkSchema] = {}
        self.header = header

    @classmethod
    def make_schema(cls, source: h5py.Group | h5py.File,  columns: Iterable[str], index: DataIndex, header: Optional[OpenCosmoHeader] = None):
        schema = DatasetSchema(header)
        for colname in columns:
            if colname not in source.keys():
                raise ValueError("Dataset source is missing some columns!")
            column_schema = ColumnSchema(colname, index, source[colname])
            schema.add_child(column_schema, colname)
        return schema

    def insert(self, child: iop.DataSchema, path: str):
        if "." in path:
            raise ValueError("Datasets do not have grandchildren!")
        return self.add_child(child, path)


    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.columns or name in self.links:
            raise ValueError(f"DatasetScheema already has a child named {name}")
        match child:
            case IdxLinkSchema() | StartSizeLinkSchema():
                self.links[name] = child
            case ColumnSchema():
                self.columns[name] = child
            case _:
                raise ValueError(f"Dataset schema cannot take children of type {type(child)}")

    def verify(self):
        if len(self.columns) == 0:
            raise ValueError("Datasets must have at least one column")

        column_lengths = set(len(c.index) for c in self.columns.values())
        if len(column_lengths) > 1:
            raise ValueError("Datasets columns must be the same length!")

        for child in chain(self.columns.values(), self.links.values()):
            child.verify()

    def allocate(self, group: h5py.File | h5py.Group):
        data_group = group.require_group("data")
        for column in self.columns.values():
            column.allocate(data_group)
        for link in self.links.values():
            link_group = group.require_group("data_linked")
            link.allocate(link_group)

    def into_writer(self):
        column_writers = {name: col.into_writer() for name, col in self.columns.items()}
        link_writers = {name: link.into_writer() for name, link in self.links.items()}
        
        return iow.DatasetWriter(column_writers, link_writers, self.header)


class ColumnSchema:
    def __init__(self, name: str, index: DataIndex, source: h5py.Dataset):
        self.name = name
        self.index = index
        self.source = source
        self.offset = 0

    def concatenate(self, *others: "ColumnSchema"):
        for other in others:
            if other.name != self.name:
                raise ValueError("Tried to combine columns with different names")
            if self.source.shape[1:] != other.source.shape[1:]:
                raise ValueError("Tried to combine columns with incompatible shapes")

        new_index = self.index.concatenate(*[o.index for o in others])
        return ColumnSchema(self.name, new_index, self.source)
        

    def add_child(self, *args, **kwargs):
        raise TypeError("Columns do not take children!")
    insert = add_child

    def set_offset(self, offset: int):
        self.offset = offset


    def verify(self):
        if len(self.index) == 0:
            raise ValueError("Columns cannot have zero length!")
        if len(self.index) > len(self.source):
            raise ValueError("The index is longer than its source!")

    def allocate(self, group: h5py.Group):
        shape = (len(self.index),) + self.source.shape[1:]
        group.create_dataset(self.name, shape, self.source.dtype, compression=COMPRESSION)

    def into_writer(self):
        return iow.ColumnWriter(self.name, self.index, self.source, self.offset)

class IdxLinkSchema:
    def __init__(self, name: str, index: DataIndex, source: h5py.Dataset):
        self.column = ColumnSchema(f"{name}_idx", index, source)

    def allocate(self, group: h5py.Group):
        return self.column.allocate(group)

    def add_child(self, *args, **kwargs):
        raise TypeError("Links do not take children!")
    insert = add_child

    def verify(self):
        return self.column.verify()

    def into_writer(self):
        return iow.IdxLinkWriter(self.column.into_writer())

class StartSizeLinkSchema:
    def __init__(self, name: str, index: DataIndex, start: h5py.Dataset, size: h5py.Dataset, link_offset: int = 0, start_offset: int = 0):
        self.name = name
        self.index = index
        self.start = ColumnSchema(f"{name}_start", index, start)
        self.size = ColumnSchema(f"{name}_size", index, size)
        self.start.set_offset(link_offset)
        self.size.set_offset(link_offset)
        self.start_offset = start_offset

    def allocate(self, group: h5py.Group):
        self.start.allocate(group)
        self.size.allocate(group)

    def add_child(self, *args, **kwargs):
        raise TypeError("Links do not take children!")
    insert = add_child

    def verify(self):
        self.start.verify()
        self.size.verify()

    def into_writer(self):
        return iow.StartSizeLinkWriter(self.start.into_writer(), self.size.into_writer())

