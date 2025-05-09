from itertools import chain
from typing import Iterable, Optional
import opencosmo.io.protocols as iop
import opencosmo.io.writers  as iow
from opencosmo.dataset.index import DataIndex
from opencosmo.header import OpenCosmoHeader

import h5py
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

class FileSchema:

    def __init__(self):
        self.children = {}

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
        self.datasets: dict[str, DatasetSchema] = {}
        self.header = header

    def insert(self, child: iop.DataSchema, path: str):
        try:
            child_name, remaining_path = path.split(".", maxsplit=1)
            if child_name not in self.datasets:
                raise ValueError(f"File schema has no child {child_name}")
            self.datasets[child_name].insert(child, remaining_path)
        except ValueError:
            self.add_child(child, path)

    def verify(self):
        if len(self.datasets) < 2:
            raise ValueError("StructCollection must have at least two children!")

        found_links = False
        for child in self.datasets.values():
            child.verify()
            if child.links:
                found_links = True
        if not found_links:
            raise ValueError("StructCollection must get at least one link!")
            

    def add_child(self, child: iop.DataSchema, name: str):
        if name in self.datasets:
            raise ValueError(f"StructCollectionSchema already has child with name {name}")
        match child:
            case DatasetSchema():
                self.datasets[name] = child
            case LinkSchema():
                raise ValueError("LinkSchemas need to be added to a DatasetSchema directly. Perhaps you meant to call insert?")
            case _: 
                raise ValueError(f"StructCollectionSchema cannot take children of type {type(child)}")


    def allocate(self, group: h5py.Group):
        if len(self.datasets) == 1:
            raise ValueError("A dataset collection cannot have only one member!")
        for ds_name, ds in self.datasets.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)

    def into_writer(self):
        dataset_writers = {key: val.into_writer() for key, val in self.datasets.items()}
        return iow.CollectionWriter(dataset_writers, self.header)

class DatasetSchema:
    def __init__(
        self,
        header: Optional[OpenCosmoHeader] = None,
    ):
        self.columns: dict[str, ColumnSchema] = {}
        self.links: dict[str, LinkSchema] = {}
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
            case LinkSchema():
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
        column_writers = [col.into_writer() for col in self.columns.values()]
        link_writers = [link.into_writer() for link in self.links.values()]
        
        return iow.DatasetWriter(column_writers, link_writers, self.header)


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
    def __init__(self, name: str, index: DataIndex, source: h5py.Dataset | tuple[h5py.Dataset, h5py.Dataset]):
        self.name = name
        self.source = source
        self.index = index

    def add_child(self, *args, **kwargs):
        raise TypeError("Links do not take children!")
    insert = add_child

    def verify(self):
        if len(self.index) == 0:
            raise ValueError("Links cannot have zero length")


    def allocate(self, group: h5py.Group):
        if isinstance(self.source, tuple):
            start_name = f"{self.name}_start"
            size_name = f"{self.name}_size"
            group.create_dataset(start_name, shape=(len(self.index,)), dtype=np.uint64)
            group.create_dataset(size_name, shape=(len(self.index,)), dtype=np.uint64)
        else:
            name = f"{self.name}_idx"
            group.create_dataset(name, shape=(len(self.index),), dtype=np.uint64)

    def into_writer(self):
        if isinstance(self.source, tuple):
            return iow.StartSizeLinkWriter(self.name, self.index, self.source[1])
        else:
            return iow.IdxLinkWriter(self.name, self.index, self.source)
        





