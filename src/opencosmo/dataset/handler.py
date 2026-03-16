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
        columns: dict[str, h5py.Dataset],
        index: DataIndex,
        metadata_columns: dict[str, h5py.Dataset],
        load_conditions: Optional[dict[str, bool]] = None,
    ):
        self.__index = index
        self.__columns = columns
        self.__metadata_columns = metadata_columns
        self.__in_memory = next(iter(columns.values())).file.driver == "core"
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

        data_columns_ = {col.name.split("/")[-1]: col for col in data_columns}
        metadata_columns_ = {col.name.split("/")[-1]: col for col in metadata_columns}
        lengths = set(
            len(col)
            for col in chain(data_columns_.values(), metadata_columns_.values())
        )
        if len(lengths) > 1:
            raise ValueError("Not all columns are the same length!")

        if index is None:
            index = from_size(lengths.pop())

        return Hdf5Handler(data_columns_, index, metadata_columns_, load_conditions)

    @property
    def in_memory(self) -> bool:
        return self.__in_memory

    @property
    def load_conditions(self) -> Optional[dict[str, bool]]:
        return self.__load_conditions

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None):
        if len(other) == 0:
            return Hdf5Handler(
                self.__columns, other, self.__metadata_columns, self.__load_conditions
            )

        if sorted is not None:
            return self.__take_sorted(other, sorted)

        new_index = take(self.__index, other)
        return Hdf5Handler(
            self.__columns, new_index, self.__metadata_columns, self.__load_conditions
        )

    def __take_sorted(self, other: DataIndex, sorted: np.ndarray):
        if get_length(sorted) != get_length(self.__index):
            raise ValueError("Sorted index has the wrong length!")
        new_indices = get_data(other, sorted)

        new_raw_index = into_array(self.__index)[new_indices]
        new_index = np.sort(new_raw_index)

        return Hdf5Handler(
            self.__columns, new_index, self.__metadata_columns, self.__load_conditions
        )

    @property
    def data(self):
        return next(iter(self.__columns.values())).parent

    @property
    def index(self):
        return self.__index

    @cached_property
    def columns(self):
        return self.__columns.keys()

    @property
    def metadata_columns(self):
        return self.__metadata_columns.keys()

    @cached_property
    def descriptions(self):
        return {
            colname: column.attrs.get("description")
            for colname, column in self.__columns.items()
        }

    def mask(self, mask):
        idx = SimpleIndex(np.where(mask)[0])
        return self.take(idx)

    def __len__(self) -> int:
        return len(next(iter(self.__columns.values())))

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
            column_writers[column_name] = ColumnWriter.from_h5_dataset(
                self.__columns[column_name],
                self.__index,
                attrs=dict(self.__columns[column_name].attrs),
            )

        data_schema = make_schema("data", FileEntry.COLUMNS, columns=column_writers)

        metadata_schema = None

        if self.metadata_columns:
            assert len(self.__metadata_columns) > 0
            metadata_writers = {}
            group_name = next(iter(self.__metadata_columns.values())).parent.name
            group_name = group_name.split("/")[-1]
            for column_name, column in self.__metadata_columns.items():
                metadata_writers[column_name] = ColumnWriter.from_h5_dataset(
                    column, self.__index, attrs=dict(column.attrs)
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
            data[colname] = get_data(self.__columns[colname], self.__index)
        # Ensure order is preserved
        return {name: data[name] for name in columns}

    def get_metadata(self, columns: Iterable[str]) -> Optional[dict[str, np.ndarray]]:
        if len(self.__metadata_columns) == 0:
            return None
        if not columns:
            columns = self.metadata_columns

        data = {}
        for colname in columns:
            data[colname] = get_data(self.__metadata_columns[colname], self.__index)

        return data

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]
