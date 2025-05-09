from typing import Iterable, Optional, Protocol, TypeVar

import h5py

import numpy as np
from opencosmo.dataset.index import DataIndex
from opencosmo.io import schemas as ios
from opencosmo.io import protocols as iop
from opencosmo.header import OpenCosmoHeader


def write_index(
    input_ds: h5py.Dataset,
    output_ds: h5py.Dataset,
    index: DataIndex,
    range_: Optional[tuple[int, int]] = None,
):
    if len(index) == 0:
        raise ValueError("No indices provided to write")
    data = index.get_data(input_ds)
    output_ds[:] = data


    attrs = input_ds.attrs
    for key in attrs.keys():
        output_ds.attrs[key] = attrs[key]

class CollectionWriter:
    def __init__(self, children: dict[str, iop.DataWriter], header: Optional[OpenCosmoHeader] = None):
        self.children = children
        self.header = header

    def write(self, file: h5py.File | h5py.Group):
        for name, dataset in self.children.items():
            dataset.write(file[name])

        if self.header is not None:
            self.header.write(file)

class FileWriter:
    def __init__(self, children: dict[str, iop.DataWriter]): 
        self.children = children

    def write(self, file: h5py.File):
        if len(self.children) == 1:
            ds = next(iter(self.children.values()))
            return ds.write(file)
        for name, dataset in self.datasets.items():
            dataset.write(file[name])


class DatasetWriter:
    def __init__(self, columns: list["ColumnWriter"], links: list = [], header: Optional[OpenCosmoHeader] = None):
        self.columns = columns
        self.header = header
        self.links = links

    def write(self, group: h5py.Group, range_: Optional[tuple[int,int]] = None):
        data_group = group["data"]
        for column in self.columns:
            dataset = data_group[column.name]
            column.write(dataset, range_)

        if self.links:
            link_group = group["data_linked"]
            for link in self.links:
                link.write(link_group)


        if self.header is not None:
            self.header.write(group)


class ColumnWriter:
    def __init__(self, name: str, index: DataIndex, source: h5py.Dataset):
        self.name = name
        self.source = source
        self.index = index

    def write(self, dataset: h5py.Dataset, range_: Optional[tuple[int,int]] = None):
        write_index(self.source, dataset, self.index, range_)
        for name, val in self.source.attrs.items():
            dataset.attrs[name] = val

class IdxLinkWriter:
    def __init__(self, name: str, index: DataIndex, source: h5py.Dataset):
        self.name = name
        self.index = index
        self.source = source

    def write(self, group: h5py.Group):
        new_idxs = np.full(len(self.index), -1)
        current_values = self.index.get_data(self.source)
        has_data = current_values > 0
        new_idxs[has_data] = np.arange(sum(has_data))
        group[f"{self.name}_idx"][:] = new_idxs


class StartSizeLinkWriter:
    def __init__(self, name: str, index: DataIndex, sizes: h5py.Dataset):
        self.name = name
        self.index = index
        self.sizes = sizes

    def write(self, group: h5py.Group):
        new_sizes = self.index.get_data(self.sizes)
        new_starts = np.insert(np.cumsum(new_sizes), 0, 0)
        new_starts = new_starts[:-1]
        group[f"{self.name}_start"][:] = new_starts
        group[f"{self.name}_size"][:] = new_sizes

        print(new_starts)
        


