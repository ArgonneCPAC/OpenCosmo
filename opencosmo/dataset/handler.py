from __future__ import annotations

from typing import Iterable, Optional, TypeVar

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.dataset.index import DataIndex
from opencosmo.spatial.tree import Tree
from opencosmo.header import OpenCosmoHeader
from opencosmo.io.schemas import DatasetSchema

try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() == 1:
        MPI = None
except ImportError:
    MPI = None



class DatasetHandler:
    """
    Handler for opencosmo.Dataset
    """

    def __init__(
        self,
        file: h5py.File,
        tree: Optional[Tree] = None,
        group_name: Optional[str] = None,
    ):
        self.__group_name = group_name
        self.__file = file
        if group_name is None:
            self.__group = file["data"]
        else:
            self.__group = file[f"{group_name}/data"]
        self.__tree = tree

    def __len__(self) -> int:
        first_column_name = next(iter(self.__group.keys()))
        return self.__group[first_column_name].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        return self.__file.close()

    def collect(self, columns: Iterable[str], index: DataIndex) -> DatasetHandler:
        if MPI is not None:
            indices = MPI.COMM_WORLD.allgather(index)
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


        return DatasetHandler(
            file,
            group_name = self.__group_name
        )
            
    def prep_write(
        self,
        index: DataIndex,
        columns: Iterable[str],
        dataset_name: str,
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

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]
