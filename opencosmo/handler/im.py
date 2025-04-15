from __future__ import annotations

from typing import Iterable, Optional

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.dataset.index import ChunkedIndex, DataIndex
from opencosmo.file import get_data_structure
from opencosmo.spatial.tree import Tree


class InMemoryHandler:
    """
    A handler for in-memory storage. All data will be loaded directly into memory.

    Reading a file is always done with OpenCosmo.read, which manages file opening
    and closing.
    """

    def __init__(
        self,
        file: h5py.File,
        tree: Optional[Tree] = None,
        group_name: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
        index: Optional[DataIndex] = None,
    ):
        if group_name is None:
            group = file["data"]
        else:
            group = file[f"{group_name}/data"]
        self.__columns = get_data_structure(group)
        if columns is not None:
            self.__columns = {n: u for n, u in self.__columns.items() if n in columns}
        self.__tree = tree
        if index is None:
            length = len(next(iter(group.values())))
            index = ChunkedIndex.from_size(length)
        self.__data = {
            colname: index.get_data(group[colname]) for colname in self.__columns
        }

    def __len__(self) -> int:
        return len(next(iter(self.__data.values())))

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        return False

    def collect(self, columns: Iterable[str], index: DataIndex) -> InMemoryHandler:
        """
        Create a new InMemoryHandler with only the specified columns and
        the specified mask applied.
        """
        new_data = {
            colname: index.get_data(self.__data[colname]) for colname in columns
        }
        mask = np.zeros(len(self), dtype=bool)
        mask = index.set_data(mask, True)
        if self.__tree is not None:
            tree = self.__tree.apply_mask(mask)
        else:
            tree = None
        return InMemoryHandler(new_data, tree)

    def write(
        self,
        file: h5py.File,
        index: DataIndex,
        columns: Iterable[str],
        dataset_name: Optional[str] = None,
    ) -> None:
        """
        Write the data in the specified columns, with the specified mask, to the file.
        """
        if dataset_name is None:
            group = file
        else:
            group = file.require_group(dataset_name)
        data_group = group.require_group("data")
        for column in columns:
            data_group.create_dataset(column, data=index.get_data(self.__data[column]))
            if self.__columns[column] is not None:
                data_group[column].attrs["unit"] = self.__columns[column]
        mask = np.zeros(len(self), dtype=bool)
        if self.__tree is not None:
            tree = self.__tree.apply_mask(mask)
            tree.write(group, dataset_name="index")

    def get_data(
        self,
        builders: dict,
        index: DataIndex,
    ) -> Column | Table:
        """
        Get data from the in-memory storage with optional masking and column
        selection.
        """

        output = {}
        for column, builder in builders.items():
            col = index.get_data(self.__data[column])
            output[column] = builder.build(Column(col, name=column))

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)
