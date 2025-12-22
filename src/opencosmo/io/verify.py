from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from opencosmo.index import DataIndex, get_data, get_length

if TYPE_CHECKING:
    import h5py
    import numpy as np
    from numpy.typing import DTypeLike

    from opencosmo.index import DataIndex


""" Columns in a schema are given by their path within the file.
"""


class ColumnCombineStrategy(Enum):
    SUM = "sum"
    CONCAT = "concat"


def verify_file(columns: dict[str, ColumnWriter]):
    if not columns:
        raise ValueError("Got no column writers to verify!")
    if any(colname.startswith("data") for colname in columns.keys()):
        # Dealing with a single, unnested dataset
        verify_dataset_data(columns)


def verify_dataset_data(schema: dict[str, ColumnWriter]):
    """
    Verify a given dataset is valid. Requiring:
    1. It has a data group
    2. It has a spatial index group
    3. If it has any metadata groups, they are the same length as the data group
    """
    data_group_columns = {}
    index_group_columns = {}
    metadata_group_columns = {}
    index_root = None
    for column_path, column_writer in schema.items():
        if "data" in column_path:
            data_group_columns[column_path] = column_writer
        elif "index" in column_path:
            index_group_columns[column_path] = column_writer
            if index_root is None:
                (prefix, index, _) = column_path.partition("index")
                index_root = prefix + index

        else:
            metadata_group_columns[column_path] = column_writer

    if not data_group_columns or not index_group_columns:
        raise ValueError("Datasets must have at least a data group and a index group")

    verify_column_group(data_group_columns)
    verify_column_group(
        index_group_columns, verify_root=index_root, verify_length_by_group=True
    )
    if metadata_group_columns:
        verify_column_group(metadata_group_columns)


def verify_column_group(
    schema: dict[str, ColumnWriter],
    verify_root: Optional[str] = None,
    verify_length_by_group=False,
):
    """
    Verify that a given data group is valid. This requires that:
    1. All column writers have the same length
    2. All columns have the same combine strategy
    3. All columns are in the same group

    By default, requires that all columns are in the same group. If "verify_root"
    is set, verifies that all columns are in "verify_root", but does not require
    they are in the same group.

    If verify_length_by_group is set, length verification is done on a per-group basis
    rather than for all columns together. This only has an effect when verify_root
    is set.
    """
    column_names = set()
    group_names = set()
    column_lengths = {}
    column_strategies = set()
    for column_path, column_writer in schema.items():
        try:
            group_name, column_name = column_path.rsplit("/", 1)
        except ValueError:
            group_name = None
            column_name = column_path
        group_names.add(group_name)
        column_names.add(column_name)
        column_lengths[column_path] = len(column_writer)
        column_strategies.add(column_writer.combine_strategy)

    all_column_lengths = set(column_lengths.values())

    if verify_root is None and len(group_names) != 1:
        raise ValueError(
            "Attempted to verify a single column group, but got columns in sepearate groups"
        )

    elif verify_root is not None and not all(
        gn.startswith(verify_root) for gn in group_names
    ):
        raise ValueError(f"Columns in this group should be relative to {verify_root}")

    if len(all_column_lengths) != 1 and not verify_length_by_group:
        raise ValueError(
            "Columns within a single group should always have the same length!"
        )
    elif verify_length_by_group:
        verify_lengths_by_groups(group_names, column_lengths)

    if len(column_strategies) != 1:
        raise ValueError(
            "Columns within a single group should always have the same combine strategy!"
        )

    return (group_names.pop(), column_lengths, column_strategies.pop())


def verify_lengths_by_groups(groups: set[str], column_lengths: dict[str, int]):
    for group_name in groups:
        columns_in_group = filter(
            lambda kv: kv[0].startswith(group_name), column_lengths.items()
        )
        lengths = set(map(lambda kv: kv[1], columns_in_group))
        if len(lengths) != 1:
            raise ValueError(
                f"Columns in group {group_name} do not have the same length!"
            )


class ColumnSource(Protocol):
    def __len__(self) -> int: ...
    @property
    def dtype(self) -> DTypeLike: ...
    @property
    def data(self) -> np.ndarray: ...


class Hdf5Source:
    def __init__(self, h5ds: h5py.Dataset, index: DataIndex):
        self.__source = h5ds
        self.__index = index

    def __len__(self):
        return get_length(self.__index)

    @property
    def dtype(self):
        return self.__source.dtype

    @property
    def data(self):
        return get_data(self.__source, self.__index)


class NumpySource:
    def __init__(self, arr: np.ndarray):
        self.__source = arr

    def __len__(self):
        return len(self.__source)

    @property
    def dtype(self):
        return self.__source.dtype

    @property
    def data(self):
        return self.__source


class ColumnWriter:
    def __init__(
        self,
        column_sources: list[ColumnSource],
        combine_strategy: ColumnCombineStrategy,
        attrs: dict[str, Any] = {},
    ):
        self.__sources = column_sources
        self.__combine_strategy = combine_strategy
        self.__attrs = attrs
        dtypes = set(map(lambda source: source.dtype, self.__sources))
        if len(dtypes) > 1:
            raise ValueError("A single column can not have multiple data types!")
        self.__dtype = dtypes.pop()

    @classmethod
    def from_h5_dataset(
        cls,
        dataset: h5py.Dataset,
        index: DataIndex,
        strategy: ColumnCombineStrategy = ColumnCombineStrategy.CONCAT,
        attrs: dict[str, Any] = {},
    ):
        source = Hdf5Source(dataset, index)
        return ColumnWriter([source], strategy, attrs)

    def __len__(self):
        return sum(len(source) for source in self.__sources)

    @property
    def combine_strategy(self) -> ColumnCombineStrategy:
        return self.__combine_strategy

    @property
    def dtype(self) -> DTypeLike:
        return self.__dtype

    @property
    def attrs(self) -> dict[str, Any]:
        return self.__attrs

    @property
    def data(self) -> np.ndarray:
        return np.concatenate([source.data for source in self.__sources])
