from pathlib import Path
from typing import Iterable, Optional, Protocol, Self

import h5py
import hdf5plugin

from opencosmo.dataset.index import DataIndex
from opencosmo.io.schema import ColumnSchema, DatasetSchema, IndexSchema, LinkSchema, FileSchema
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

def make_dataset_schema(
    input_dataset_group: h5py.Group,
    column_names: Iterable[str],
    n_elements: int,
    link_schemas: list[LinkSchema] = [],
    index_schemas: list[IndexSchema] = [],
    additional_columns: list[ColumnSchema] = [],
    include_header: bool = True,
) -> DatasetSchema:
    column_schemas = []
    input_column_names = set(input_dataset_group.keys())
    if not set(column_names).issubset(input_column_names):
        raise ValueError(
            "Dataset schema recieved columns that are not in the original dataset!"
        )

    for name in column_names:
        col = input_dataset_group[name]
        shape = col.shape
        new_shape = (n_elements,) + shape[1:]
        column_schemas.append(ColumnSchema(name, new_shape, col.dtype))

    column_schemas.extend(additional_columns)
    return DatasetSchema(column_schemas, link_schemas, index_schemas, include_header)

class FileWriter:
    def __init__(self):
        self.schema = FileSchema()
        self.datasets: dict[str, DatasetWriter] = {}

    def add_dataset(self, name: str, schema: DatasetSchema, source: h5py.Group, index: DataIndex, header: Optional[OpenCosmoHeader] = None):
        self.schema.add_dataset(name, schema)
        self.datasets[name] = DatasetWriter(schema, source, index, header)

    def allocate(self, file: h5py.File):
        return self.schema.allocate(file)

    def write(self, file: h5py.File):
        if len(self.datasets) == 1:
            ds = next(iter(self.datasets.values()))
            ds.header.write(file)
            ds.header = None
            return ds.write(file)
        for name, dataset in self.datasets.items():
            dataset.write(file[name])

class DatasetWriter:
    def __init__(self, schema: DatasetSchema, source: h5py.Group, index: DataIndex, header: Optional[OpenCosmoHeader] = None):
        self.source = source
        self.schema = schema
        self.index = index
        self.columns = [ColumnWriter(s, index, source[s.name]) for s in self.schema.columns]
        self.header =  header

    def write(self, group: h5py.Group, range_: Optional[tuple[int,int]] = None):
        data_group = group["data"]
        for column in self.columns:
            dataset = data_group[column.schema.name]
            column.write(dataset, range_)

        for name, val in self.source.attrs.items():
            group.attrs[name] = val
        if self.header is not None:
            self.header.write(group)


class ColumnWriter:
    def __init__(self, schema: ColumnSchema, index: DataIndex, source: h5py.Dataset):
        self.schema = schema
        self.source = source
        self.index = index

    def write(self, dataset: h5py.Dataset, range_: Optional[tuple[int,int]] = None):
        write_index(self.source, dataset, self.index, range_)
        for name, val in self.source.attrs.items():
            dataset.attrs[name] = val
