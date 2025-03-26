from __future__ import annotations

from typing import Iterable, Optional

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.dataset.column import ColumnBuilder
from opencosmo.file import get_data_structure
from opencosmo.handler import InMemoryHandler
from opencosmo.spatial.tree import Tree


class OutOfMemoryHandler:
    """
    A handler for data that will not be stored in memory. Data will remain on
    disk until needed

    """

    def __init__(self, file: h5py.File, tree: Tree, group_name: Optional[str] = None):
        self.__group_name = group_name
        self.__file = file
        if group_name is None:
            self.__group = file["data"]
        else:
            self.__group = file[f"{group_name}/data"]
        self.__columns = get_data_structure(self.__group)
        self.__tree = tree

    def __len__(self) -> int:
        return self.__group[next(iter(self.__columns))].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        self.__columns = None
        return self.__file.close()

    def collect(self, columns: Iterable[str], indices: np.ndarray) -> InMemoryHandler:
        file_path = self.__file.filename
        if len(indices) == len(self):
            tree = self.__tree
        else:
            mask = np.zeros(len(self), dtype=bool)
            mask[indices] = True
            tree = self.__tree.apply_mask(mask)

        with h5py.File(file_path, "r") as file:
            return InMemoryHandler(
                file,
                tree,
                group_name=self.__group_name,
                columns=columns,
                indices=indices,
            )

    def write(
        self,
        file: h5py.File,
        indices: np.ndarray,
        columns: Iterable[str],
        dataset_name: Optional[str] = None,
    ) -> None:
        if self.__group is None:
            raise ValueError("This file has already been closed")
        if dataset_name is None:
            group = file
        else:
            group = file.require_group(dataset_name)

        data_group = group.create_group("data")
        for column in columns:
            data = self.__group[column][()]
            data = data[indices]
            data_group.create_dataset(column, data=data)
            if self.__columns[column] is not None:
                data_group[column].attrs["unit"] = self.__columns[column]
        tree_mask = np.zeros(len(self), dtype=bool)
        tree_mask[indices] = True
        tree = self.__tree.apply_mask(tree_mask)
        tree.write(group)

    def get_data(self, builders: dict, indices: np.ndarray) -> Column | Table:
        """ """
        if self.__group is None:
            raise ValueError("This file has already been closed")
        output = {}
        start_idx = indices[0]
        end_idx = indices[-1] + 1
        for column, builder in builders.items():
            data = self.__group[column][start_idx:end_idx]
            data = data[indices - start_idx]

            col = Column(data, name=column)
            output[column] = builder.build(col)

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)

    def get_range(
        self,
        start: int,
        end: int,
        builders: dict[str, ColumnBuilder],
        indices: np.ndarray,
    ) -> dict[str, tuple[float, float]]:
        if self.__group is None:
            raise ValueError("This file has already been closed")
        output = {}
        start_idx = indices[start]
        end_idx = indices[end] + 1
        for column, builder in builders.items():
            data = self.__group[column][start_idx:end_idx]
            data = data[indices[start:end]]
            col = Column(data, name=column)
            output[column] = builder.build(col)

        return Table(output)

    def take_indices(self, n: int, strategy: str, indices: np.ndarray) -> np.ndarray:
        if n > (length := len(indices)):
            raise ValueError(
                f"Requested {n} elements, but only {length} are available."
            )

        if strategy == "start":
            return indices[:n]
        elif strategy == "end":
            return indices[-n:]
        elif strategy == "random":
            return np.sort(np.random.choice(indices, n, replace=False))
        else:
            raise ValueError(
                "Strategy for `take` must be one of 'start', 'end', or 'random'"
            )
