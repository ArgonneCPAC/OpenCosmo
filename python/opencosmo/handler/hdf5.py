from __future__ import annotations

from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Iterable, Optional

import h5py
import numpy as np
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import (
    ColumnWriter,
)

from opencosmo.index import (
    SimpleIndex,
    from_size,
    get_data,
    get_length,
    into_array,
    take,
)

if TYPE_CHECKING:
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.schema import Schema

    from opencosmo.index import DataIndex


ColumnSpec = tuple[h5py.Dataset, bool]


class Hdf5Handler:
    """
    Handler for opencosmo.Dataset
    """

    def __init__(
        self,
        columns: dict[str, ColumnSpec],
        index: DataIndex,
        load_conditions: Optional[dict[str, bool]] = None,
    ):
        self.__index = index
        self.__columns = columns
        self.__in_memory = next(iter(columns.values()))[0].file.driver == "core"
        self.__load_conditions = load_conditions

    @classmethod
    def from_columns(
        cls,
        columns: list[h5py.Dataset],
        index: Optional[DataIndex] = None,
        metadata_group: Optional[str] = None,
        load_conditions: Optional[dict[str, bool]] = None,
    ):
        data_columns = filter(lambda col: col.name.split("/")[-2] == "data", columns)
        metadata_columns: Iterable[h5py.Dataset] = []
        if metadata_group:
            metadata_columns = filter(
                lambda col: col.name.split("/")[-2] == metadata_group, columns
            )

        data_columns_ = {col.name.split("/")[-1]: (col, True) for col in data_columns}
        metadata_columns_ = {
            col.name.split("/")[-1]: (col, False) for col in metadata_columns
        }
        lengths = set(
            len(col[0])
            for col in chain(data_columns_.values(), metadata_columns_.values())
        )
        if len(lengths) > 1:
            raise ValueError("Not all columns are the same length!")

        if index is None:
            index = from_size(lengths.pop())

        return Hdf5Handler(data_columns_ | metadata_columns_, index, load_conditions)

    def __len__(self):
        return get_length(self.__index)

    @property
    def in_memory(self) -> bool:
        return self.__in_memory

    @property
    def load_conditions(self) -> Optional[dict[str, bool]]:
        return self.__load_conditions

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None):
        if len(other) == 0:
            return Hdf5Handler(self.__columns, other, self.__load_conditions)

        if sorted is not None:
            return self.__take_sorted(other, sorted)

        new_index = take(self.__index, other)
        return Hdf5Handler(self.__columns, new_index, self.__load_conditions)

    def __take_sorted(self, other: DataIndex, sorted: np.ndarray):
        if get_length(sorted) != get_length(self.__index):
            raise ValueError("Sorted index has the wrong length!")
        new_indices = get_data(other, sorted)

        new_raw_index = into_array(self.__index)[new_indices]
        new_index = np.sort(new_raw_index)

        return Hdf5Handler(self.__columns, new_index, self.__load_conditions)

    @property
    def data(self):
        return next(iter(self.__columns.values()))[0].parent

    @property
    def index(self):
        return self.__index

    @cached_property
    def columns(self):
        return [
            colname for colname in self.__columns.keys() if self.__columns[colname][1]
        ]

    @property
    def metadata_columns(self):
        return [
            colname
            for colname in self.__columns.keys()
            if not self.__columns[colname][1]
        ]

    @cached_property
    def descriptions(self):
        return {
            colname: column[0].attrs.get("description")
            for colname, column in self.__columns.items()
            if column[1]
        }

    def mask(self, mask):
        idx = SimpleIndex(np.where(mask)[0])
        return self.take(idx)

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__columns = None
        return self.__file.close()

    def make_schema(
        self,
        columns: Iterable[str],
        header: Optional[OpenCosmoHeader] = None,
    ) -> tuple[Schema, Optional[Schema]]:
        column_writers = {}
        for column_name in columns:
            column = self.__columns[column_name]
            if not column[1]:
                continue
            column_writers[column_name] = ColumnWriter.from_h5_dataset(
                column[0],
                self.__index,
                attrs=dict(column[0].attrs),
            )

        data_schema = make_schema("data", FileEntry.COLUMNS, columns=column_writers)

        metadata_columns = {
            name: column for name, column in self.__columns.items() if not column[1]
        }
        if not metadata_columns:
            metadata_schema = make_schema("metadata", FileEntry.EMPTY)
            return data_schema, metadata_schema

        metadata_writers = {}
        group_name = next(iter(metadata_columns.values()))[0].parent.name
        group_name = group_name.split("/")[-1]
        for column_name, column in metadata_columns.items():
            metadata_writers[column_name] = ColumnWriter.from_h5_dataset(
                column[0], self.__index, attrs=dict(column[0].attrs)
            )
        metadata_schema = make_schema(
            group_name, FileEntry.COLUMNS, columns=metadata_writers
        )
        return data_schema, metadata_schema

    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        """ """
        if self.__columns is None:
            raise ValueError("This file has already been closed")
        data = {}

        for colname in columns:
            data[colname] = get_data(self.__columns[colname][0], self.__index)
        # Ensure order is preserved
        return {name: data[name] for name in columns}

    def get_metadata(self, columns: Iterable[str]) -> Optional[dict[str, np.ndarray]]:
        metadata_columns = {
            name: col[0] for name, col in self.__columns.items() if not col[1]
        }
        if len(metadata_columns) == 0:
            return None
        if not columns:
            columns = metadata_columns.keys()

        data = {}
        for colname in columns:
            data[colname] = get_data(metadata_columns[colname], self.__index)

        return data

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]
