from __future__ import annotations

from typing import Iterable, Optional

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.dataset.index import DataIndex
from opencosmo.spatial.tree import Tree
from opencosmo.utils import write_index
from opencosmo.io.schema import DatasetSchema
from opencosmo.io.writer import FileWriter, make_dataset_schema
from opencosmo.header import OpenCosmoHeader



class OutOfMemoryHandler:
    """
    A handler for data that will not be stored in memory. Data will remain on
    disk until needed
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

    def collect(self, columns: Iterable[str], index: DataIndex) -> OutOfMemoryHandler:
        tree: Optional[Tree] = None
        if self.__tree is not None and len(index) == len(self):
            mask = np.zeros(len(self), dtype=bool)
            mask = index.set_data(mask, True)
            tree = self.__tree.apply_mask(mask)

        else:
            tree = self.__tree

        file: h5py.File = h5py.File.in_memory()
        group = file.require_group("data")
        for colname in columns:
            dataset = self.__group[colname]
            data = index.get_data(dataset)
            group.create_dataset(colname, data=data)
            for name, value in dataset.attrs.items():
                group[colname].attrs[name] = value



        return OutOfMemoryHandler(
            file,
            tree,
        )

            
    def prep_write(
        self,
        writer: FileWriter,
        index: DataIndex,
        columns: Iterable[str],
        dataset_name: str,
        header: Optional[OpenCosmoHeader] = None,
    ) -> None:
        schema = make_dataset_schema(self.__group, columns, len(index))
        writer.add_dataset(dataset_name, schema, self.__group, index, header)
        return writer
        




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
