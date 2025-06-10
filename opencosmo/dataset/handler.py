from __future__ import annotations

from functools import reduce
from typing import Iterable, Optional, Protocol

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.header import OpenCosmoHeader
from opencosmo.index import DataIndex
from opencosmo.io.schemas import DatasetSchema
from opencosmo.mpi import get_comm_world


class DatasetHandler(Protocol):
    def __len__(self) -> int: ...
    def get_data(self, builders: dict, index: DataIndex) -> Column | Table: ...
    def make_schema(
        self,
        index: DataIndex,
        columns: Iterable[str],
        header: Optional[OpenCosmoHeader],
    ) -> DatasetSchema: ...


class MultiDatasetHandler:
    """
    Handler for a dataset that is logically the vertical concatenation of
    several datasets. In other words, all datasets have the same columns.
    """

    def __init__(self, files: Iterable[h5py.File], group_name: Optional[str] = None):
        self.__group_name = "data" if group_name is None else group_name
        self.__groups = [f[self.__group_name] for f in files]
        colnames = [set(g.keys()) for g in self.__groups]
        if colnames[0] != reduce(lambda f, s: f.union(s), colnames):
            raise ValueError("Not all datasets have the same columns!")

    def __len__(self):
        first_column_name = next(iter(self.__groups[0].keys()))
        return sum(len(g[first_column_name]) for g in self.__groups)

    def n_datasets(self):
        return len(self.groups)

    def get_data(self, builders: dict, index: DataIndex) -> Column | Table:
        """ """
        output = {}
        first_column_name = next(iter(self.__groups[0].keys()))
        sizes = np.array([len(g[first_column_name]) for g in self.__groups])
        starts = np.insert(np.cumsum(sizes), 0, 0)
        indices = [index.project(start, size) for start, size in zip(starts, sizes)]
        for column, builder in builders.items():
            values = [
                idx.get_data(group[column])
                for idx, group in zip(indices, self.__groups)
            ]
            data = np.concat(values)
            output[column] = builder.build(Column(data))

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)


class SingleDatasetHandler:
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

    def __len__(self) -> int:
        first_column_name = next(iter(self.__group.keys()))
        return self.__group[first_column_name].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        return self.__file.close()

    def collect(self, columns: Iterable[str], index: DataIndex) -> SingleDatasetHandler:
        if (comm := get_comm_world()) is not None:
            indices = comm.allgather(index)
            new_index = indices[0].concatenate(*indices[1:])
        else:
            new_index = index
        file: h5py.File = h5py.File.in_memory()
        group = file.require_group("data")
        for colname in columns:
            dataset = self.__group[colname]
            data = new_index.get_data(dataset)
            group.create_dataset(colname, data=data)
            for name, value in dataset.attrs.items():
                group[colname].attrs[name] = value

        return SingleDatasetHandler(file, group_name=self.__group_name)

    def make_schema(
        self,
        index: DataIndex,
        columns: Iterable[str],
        header: Optional[OpenCosmoHeader] = None,
    ) -> DatasetSchema:
        return DatasetSchema.make_schema(self.__group, columns, index, header)

    def get_data(self, builders: dict, index: DataIndex) -> Column | Table:
        """ """
        if self.__group is None:
            raise ValueError("This file has already been closed")
        output = {}
        for column, builder in builders.items():
            col = Column(index.get_data(self.__group[column]))
            output[column] = builder.build(col)

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)
