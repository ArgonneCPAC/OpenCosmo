from __future__ import annotations

from typing import Iterable, Optional

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.file import get_data_structure
from opencosmo.handler import InMemoryHandler
from opencosmo.spatial.tree import Tree


class OutOfMemoryHandler:
    """
    A handler for data that will not be stored in memory. Data will remain on
    disk until needed

    """

    def __init__(self, file: h5py.File, tree: Tree, group: Optional[str] = None):
        self.__group_name = group
        self.__file = file
        if group is None:
            self.__group = file["data"]
        else:
            self.__group = file[f"{group}/data"]
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

    def collect(self, columns: Iterable[str], mask: np.ndarray) -> InMemoryHandler:
        file_path = self.__file.filename
        if np.all(mask):
            tree = self.__tree
            output_mask = None
        else:
            output_mask = mask
            tree = self.__tree.apply_mask(mask)

        with h5py.File(file_path, "r") as file:
            return InMemoryHandler(
                file,
                tree,
                group_name=self.__group_name,
                columns=columns,
                mask=output_mask,
            )

    def write(
        self,
        file: h5py.File,
        mask: np.ndarray,
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
            data = data[mask]
            data_group.create_dataset(column, data=data)
            if self.__columns[column] is not None:
                data_group[column].attrs["unit"] = self.__columns[column]
        tree = self.__tree.apply_mask(mask)
        tree.write(group)

    def get_data(
        self, builders: dict = {}, mask: Optional[np.ndarray] = None
    ) -> Column | Table:
        """ """
        if self.__group is None:
            raise ValueError("This file has already been closed")
        output = {}
        for column, builder in builders.items():
            data = self.__group[column][()]
            if mask is not None:
                data = data[mask]

            col = Column(data, name=column)
            output[column] = builder.build(col)

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)

    def take_mask(self, n: int, strategy: str, mask: np.ndarray) -> np.ndarray:
        if n > (length := np.sum(mask)):
            raise ValueError(
                f"Requested {n} elements, but only {length} are available."
            )

        indices = np.where(mask)[0]
        new_mask = np.zeros_like(mask, dtype=bool)

        if strategy == "start":
            new_mask[indices[:n]] = True
        elif strategy == "end":
            new_mask[indices[-n:]] = True
        elif strategy == "random":
            new_mask[np.random.choice(indices, n, replace=False)] = True
        else:
            raise ValueError(
                "Strategy for `take` must be one of 'start', 'end', or 'random'"
            )

        return new_mask
