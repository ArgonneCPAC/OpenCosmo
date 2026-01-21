from __future__ import annotations

from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

from opencosmo.index import (
    SimpleIndex,
    from_size,
    get_data,
    get_length,
    into_array,
    take,
)
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import (
    ColumnWriter,
)

if TYPE_CHECKING:
    import h5py

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema


class Hdf5Handler:
    """
    Handler for opencosmo.Dataset
    """

    def __init__(
        self,
        group: h5py.Group,
        index: DataIndex,
        metadata_group: Optional[h5py.Group],
        load_conditions: Optional[dict[str, bool]] = None,
    ):
        self.__index = index
        self.__data_group = group
        self.__metadata_group = metadata_group
        self.__in_memory = group.file.driver == "core"
        self.__load_conditions = load_conditions

    @classmethod
    def from_group(
        cls,
        group: h5py.Group,
        index: Optional[DataIndex] = None,
        metadata_group: Optional[h5py.Group] = None,
        load_conditions: Optional[dict[str, bool]] = None,
    ):
        if not group.name.endswith("data"):
            raise ValueError("Expected a data group")
        lengths = set(len(ds) for ds in group.values())
        if len(lengths) > 1:
            raise ValueError("Not all columns are the same length!")

        if index is None:
            index = from_size(lengths.pop())

        colnames = group.keys()
        if metadata_group is not None:
            colnames = chain(colnames, metadata_group.keys())

        return Hdf5Handler(group, index, metadata_group, load_conditions)

    @property
    def in_memory(self) -> bool:
        return self.__in_memory

    @property
    def load_conditions(self) -> Optional[dict[str, bool]]:
        return self.__load_conditions

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None):
        if len(other) == 0:
            return Hdf5Handler(
                self.__data_group, other, self.__metadata_group, self.__load_conditions
            )

        if sorted is not None:
            return self.__take_sorted(other, sorted)

        new_index = take(self.__index, other)
        return Hdf5Handler(
            self.__data_group, new_index, self.__metadata_group, self.__load_conditions
        )

    def __take_sorted(self, other: DataIndex, sorted: np.ndarray):
        if get_length(sorted) != get_length(self.__index):
            raise ValueError("Sorted index has the wrong length!")
        new_indices = get_data(other, sorted)

        new_raw_index = into_array(self.__index)[new_indices]
        new_index = np.sort(new_raw_index)

        return Hdf5Handler(
            self.__data_group, new_index, self.__metadata_group, self.__load_conditions
        )

    @property
    def data(self):
        return self.__data_group

    @property
    def index(self):
        return self.__index

    @property
    def columns(self):
        return self.__data_group.keys()

    @property
    def metadata_columns(self):
        if self.__metadata_group is None:
            return []
        return self.__metadata_group.keys()

    @cached_property
    def descriptions(self):
        return {
            colname: column.attrs.get("description")
            for colname, column in self.__data_group.items()
        }

    def mask(self, mask):
        idx = SimpleIndex(np.where(mask)[0])
        return self.take(idx)

    def __len__(self) -> int:
        first_column_name = next(iter(self.__data_group.keys()))
        return self.__data_group[first_column_name].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__data_group = None
        return self.__file.close()

    def make_schema(
        self,
        columns: Iterable[str],
        header: Optional[OpenCosmoHeader] = None,
    ) -> tuple[Schema, Optional[Schema]]:
        column_writers = {}
        for column_name in columns:
            column_writers[column_name] = ColumnWriter.from_h5_dataset(
                self.__data_group[column_name],
                self.__index,
                attrs=dict(self.__data_group[column_name].attrs),
            )

        data_schema = make_schema("data", FileEntry.COLUMNS, columns=column_writers)

        metadata_schema = None

        if self.metadata_columns:
            assert self.__metadata_group is not None
            group_name = self.__metadata_group.name.split("/")[-1]
            metadata_writers = {}
            for column_name in self.metadata_columns:
                metadata_writers[column_name] = ColumnWriter.from_h5_dataset(
                    self.__metadata_group[column_name],
                    self.__index,
                    attrs=dict(self.__metadata_group[column_name].attrs),
                )
            metadata_schema = make_schema(
                group_name, FileEntry.COLUMNS, columns=metadata_writers
            )
        return data_schema, metadata_schema

    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        """ """
        if self.__data_group is None:
            raise ValueError("This file has already been closed")
        data = {}

        for colname in columns:
            data[colname] = get_data(self.__data_group[colname], self.__index)
        # Ensure order is preserved
        return {name: data[name] for name in columns}

    def get_metadata(self, columns: Iterable[str]) -> Optional[dict[str, np.ndarray]]:
        if self.__metadata_group is None:
            return None
        if not columns:
            columns = self.metadata_columns

        data = {}
        for colname in columns:
            data[colname] = get_data(self.__metadata_group[colname], self.__index)

        return data

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]
