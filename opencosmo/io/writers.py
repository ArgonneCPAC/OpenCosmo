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
    def __init__(self, columns: list["ColumnWriter"], header: Optional[OpenCosmoHeader] = None):
        self.columns = columns
        self.header = header

    def write(self, group: h5py.Group, range_: Optional[tuple[int,int]] = None):
        data_group = group["data"]
        for column in self.columns:
            dataset = data_group[column.name]
            column.write(dataset, range_)

        if self.header is not None:
            print("WRITING HEADER")
            print(group)
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
