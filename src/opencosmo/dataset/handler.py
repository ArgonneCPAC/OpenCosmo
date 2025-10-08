from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

from opencosmo.index import ChunkedIndex, SimpleIndex
from opencosmo.index.take import take
from opencosmo.io.schemas import DatasetSchema
from opencosmo.mpi import get_comm_world

if TYPE_CHECKING:
    import h5py

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex


class Hdf5Handler:
    """
    Handler for opencosmo.Dataset
    """

    def __init__(
        self,
        group: h5py.Group,
        index: DataIndex,
    ):
        self.__index = index
        self.__group = group

    @classmethod
    def from_group(cls, group: h5py.Group, index: Optional[DataIndex] = None):
        if not group.name.endswith("data"):
            raise ValueError("Expected a data group")
        lengths = set(len(ds) for ds in group.values())
        if len(lengths) > 1:
            raise ValueError("Not all columns are the same length!")

        if index is None:
            index = ChunkedIndex.from_size(lengths.pop())

        return Hdf5Handler(group, index)

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None):
        if len(other) == 0:
            return Hdf5Handler(self.__group, other)

        if sorted is not None:
            return self.__take_sorted(other, sorted)
        new_index = take(self.__index, other)
        return Hdf5Handler(self.__group, new_index)

    def __take_sorted(self, other: DataIndex, sorted: np.ndarray):
        if len(sorted) != len(self.__index):
            raise ValueError("Sorted index has the wrong length!")
        new_indices = other.get_data(sorted)
        new_indices = self.__index.into_array()[new_indices]
        new_index = SimpleIndex(np.sort(new_indices))

        return Hdf5Handler(self.__group, new_index)

    @property
    def data(self):
        return self.__group

    @property
    def index(self):
        return self.__index

    @property
    def columns(self):
        return self.__group.keys()

    @cached_property
    def descriptions(self):
        return {
            colname: column.attrs.get("description")
            for colname, column in self.__group.items()
        }

    def mask(self, mask):
        idx = SimpleIndex(np.where(mask)[0])
        return self.take(idx)

    def __len__(self) -> int:
        first_column_name = next(iter(self.__group.keys()))
        return self.__group[first_column_name].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        return self.__file.close()

    def prep_write(
        self,
        columns: Iterable[str],
        header: Optional[OpenCosmoHeader] = None,
    ) -> DatasetSchema:
        return DatasetSchema.make_schema(self.__group, columns, self.__index, header)

    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        """ """
        if self.__group is None:
            raise ValueError("This file has already been closed")
        data = {}
        for colname in columns:
            data[colname] = self.__index.get_data(self.__group[colname])

        return data

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]
