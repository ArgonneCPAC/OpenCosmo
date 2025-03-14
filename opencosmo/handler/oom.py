from __future__ import annotations

from typing import Iterable, Optional

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.handler import InMemoryHandler
from opencosmo.spatial.tree import Tree


class OutOfMemoryHandler:
    """
    A handler for in-memory storage. Data will remain on disk until needed

    """

    def __init__(self, file: h5py.File, tree: Tree, group: str = "data"):
        self.__file = file
        self.__group = file[group]
        self.__columns = list(self.__group.keys())
        self.__tree = tree

    def __len__(self) -> int:
        return self.__group[self.__columns[0]].size

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        self.__columns = None
        return self.__file.close()

    def collect(self, columns: Iterable[str], mask: np.ndarray) -> InMemoryHandler:
        file_path = self.__file.filename
        tree = self.__tree.apply_mask(mask)
        with h5py.File(file_path, "r") as file:
            return InMemoryHandler(file, columns=columns, mask=mask, tree=tree)

    def write(
        self,
        file: h5py.File,
        mask: np.ndarray,
        columns: Iterable[str],
        dataset_name="data",
    ) -> None:
        group = file.require_group(dataset_name)
        for column in columns:
            data = self.__group[column][mask]
            group.create_dataset(column, data=data)
        tree = self.__tree.apply_mask(mask)
        tree.write(file)

    def get_data(
        self, builders: dict = {}, mask: Optional[np.ndarray] = None
    ) -> Column | Table:
        """ """
        if self.__group is None:
            raise ValueError("This file has already been closed")
        output = {}
        for column, builder in builders.items():
            if mask is None:
                data = self.__group[column][()]
            else:
                data = self.__group[column][mask]

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
