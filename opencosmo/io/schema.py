from typing import Iterable

import h5py
import numpy as np
from numpy.typing import DTypeLike

ColumnShape = tuple[int, ...]


def validate_shapes(
    columns: Iterable["ColumnSchema"], links: Iterable["LinkSchema"] = []
):
    """
    Datasets should be representable as tables, so the first dimension
    of all shapes must be the same.
    """
    columns = list(columns)
    links = list(links)
    n_rows = set(c.shape for c in columns)
    if len(n_rows) > 1:
        raise ValueError("All columns in a dataset must have the same length!")

    if len(links) > 0:
        validate_link_schemas(len(columns), links)


def validate_link_schemas(n_cols: int, links: Iterable["LinkSchema"]):
    if any(link.n > n_cols for link in links):
        raise ValueError("All links must have the same length as the dataset!")


def verify_file_schema(datasets: Iterable["DatasetSchema"], has_header: bool = False):
    all_datasets_have_header = all(ds.has_header for ds in datasets)
    if not has_header and not all_datasets_have_header:
        raise ValueError(
            "Either the top-level file must have a header, or all datsets must"
        )


class FileSchema:
    def __init__(self, datasets: dict[str, "DatasetSchema"], has_header: bool = False):
        self.datasets = datasets
        self.has_header = has_header
        verify_file_schema(self.datasets.values(), self.has_header)

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


class DatasetSchema:
    def __init__(
        self,
        columns: Iterable["ColumnSchema"],
        links: Iterable["LinkSchema"] = [],
        indices: Iterable["IndexSchema"] = [],
        has_header: bool = False,
    ):
        self.columns = list(columns)
        self.has_header = has_header
        self.links = links
        self.indices = indices
        if len(set(c.name for c in self.columns)) != len(self.columns):
            raise ValueError("Column names must be unique!")
        validate_shapes(self.columns, self.links)

    def allocate(self, group: h5py.File | h5py.Group):
        data_group = group.require_group("data")
        for column in self.columns:
            column.allocate(data_group)
        for link in self.links:
            link.allocate(group)
        for index in self.indices:
            index.allocate(group)


class ColumnSchema:
    def __init__(self, name: str, shape: ColumnShape, dtype: DTypeLike):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def allocate(self, group: h5py.Group):
        group.require_dataset(self.name, self.shape, self.dtype)


class LinkSchema:
    def __init__(self, name: str, n: int, has_sizes: bool = False):
        self.n = n
        self.has_sizes = has_sizes
        self.name = name

    def allocate(self, group: h5py.Group):
        if self.has_sizes:
            start_name = f"{self.name}_start"
            size_name = f"{self.name}_size"
            group.create_dataset(start_name, shape=(self.n,), dtype=np.uint64)
            group.create_dataset(size_name, shape=(self.n,), dtype=np.uint64)
        else:
            name = f"{self.name}_idx"
            group.create_dataset(name, shape=(self.n,), dtype=np.uint64)


class IndexSchema:
    def __init__(self, levels: list[int]):
        self.levels = levels

    def allocate(self, group: h5py.Group):
        for i, level_size in enumerate(self.levels):
            level_group = group.require_group(f"level_{i}")
            level_group.require_dataset("start", (level_size,), np.uint64)
            level_group.require_dataset("size", (level_size,), np.uint64)
