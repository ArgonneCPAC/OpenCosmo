from __future__ import annotations

from functools import cached_property
from typing import Iterable, Optional

import h5py
import numpy as np

from opencosmo.header import OpenCosmoHeader
from opencosmo.index import DataIndex
from opencosmo.io.schemas import DatasetSchema
from opencosmo.mpi import get_comm_world


class DatasetHandler:
    """
    Handler for opencosmo.Dataset
    """

    def __init__(
        self,
        file: h5py.File,
        group_name: Optional[str] = None,
    ):
        self.__group_name = group_name
        self.__file = file
        if group_name is None:
            self.__group = file["data"]
        else:
            self.__group = file[f"{group_name}/data"]

    @property
    def data(self):
        return self.__group

    @property
    def columns(self):
        return self.__group.keys()

    @cached_property
    def descriptions(self):
        return {
            colname: column.attrs.get("description")
            for colname, column in self.__group.items()
        }

    def __len__(self) -> int:
        first_column_name = next(iter(self.__group.keys()))
        return self.__group[first_column_name].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        return self.__file.close()

    def collect(self, columns: Iterable[str], index: DataIndex) -> DatasetHandler:
        if (comm := get_comm_world()) is not None:
            indices = comm.allgather(index)
            new_index = indices[0].concatenate(*indices[1:])
        else:
            new_index = index
        file: h5py.File = h5py.File.in_memory()
        group = file.require_group("data")
        for colname in columns:
            if colname not in self.__group.keys():
                continue
            dataset = self.__group[colname]
            data = new_index.get_data(dataset)
            group.create_dataset(colname, data=data)
            for name, value in dataset.attrs.items():
                group[colname].attrs[name] = value

        return DatasetHandler(file, group_name=self.__group_name)

    def prep_write(
        self,
        index: DataIndex,
        columns: Iterable[str],
        header: Optional[OpenCosmoHeader] = None,
    ) -> DatasetSchema:
        return DatasetSchema.make_schema(self.__group, columns, index, header)

    def get_data(
        self, columns: Iterable[str], index: DataIndex
    ) -> dict[str, np.ndarray]:
        """ """
        if self.__group is None:
            raise ValueError("This file has already been closed")
        data = {}
        for colname in columns:
            data[colname] = index.get_data(self.__group[colname])

        return data

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]
