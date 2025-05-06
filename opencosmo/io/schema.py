from typing import Iterable, Union

import h5py
import numpy as np
from numpy.typing import DTypeLike

ColumnShape = tuple[int, ...]


def validate_shapes(
    columns: Iterable["ColumnSchema"]
):
    """
    Datasets should be representable as tables, so the first dimension
    of all shapes must be the same.
    """
    columns = list(columns)
    n_rows = set(c.shape for c in columns)
    if len(n_rows) > 1:
        raise ValueError("All columns in a dataset must have the same length!")



class FileSchema:
    def __init__(self):
        self.datasets = {}


    def add_dataset(self, name: str, schema: Union["DatasetSchema", "CollectionSchema"]):
        if name in self.datasets:
            raise ValueError(f"Writer already has a dataset with name {name}")
        self.datasets[name] = schema

    def allocate(self, group: h5py.File):
        if not isinstance(group, h5py.File):
            raise ValueError(
                "File Schema allocation must be done at the top level of a h5py file!"
            )
        if len(self.datasets) == 1:
            ds = next(iter(self.datasets.values()))
            return ds.allocate(group)

        for ds_name, ds in self.datasets.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)

class CollectionSchema:
    def __init__(self):
        self.datasets = {}

    def add_dataset(self, name: str, schema: "DatasetSchema"):
        self.datasets[name] = schema

    def allocate(self, group: h5py.Group):
        if len(self.datasets) == 1:
            raise ValueError("A dataset collection cannot have only one member!")
        for ds_name, ds in self.datasets.items():
            ds_group = group.require_group(ds_name)
            ds.allocate(ds_group)



class DatasetSchema:
    def __init__(
        self,
        columns: Iterable["ColumnSchema"],
        indices: Iterable["IndexSchema"] = [],
        has_header: bool = False,
    ):
        self.columns = list(columns)
        self.has_header = has_header
        self.indices = indices
        if len(set(c.name for c in self.columns)) != len(self.columns):
            raise ValueError("Column names must be unique!")
        validate_shapes(self.columns)

    def allocate(self, group: h5py.File | h5py.Group):
        data_group = group.require_group("data")
        for column in self.columns:
            column.allocate(data_group)
        for index in self.indices:
            index.allocate(group)


class ColumnSchema:
    def __init__(self, name: str, shape: ColumnShape, dtype: DTypeLike):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def allocate(self, group: h5py.Group):
        group.require_dataset(self.name, self.shape, self.dtype)


class IndexSchema:
    def __init__(self, levels: list[int]):
        self.levels = levels

    def allocate(self, group: h5py.Group):
        for i, level_size in enumerate(self.levels):
            level_group = group.require_group(f"level_{i}")
            level_group.require_dataset("start", (level_size,), np.uint64)
            level_group.require_dataset("size", (level_size,), np.uint64)
