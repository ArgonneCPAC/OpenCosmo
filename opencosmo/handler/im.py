from __future__ import annotations
from typing import Iterable, Optional

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore


class InMemoryHandler:
    """
    A handler for in-memory storage. All data will be loaded directly into memory.

    Reading a file is always done with OpenCosmo.read, which manages file opening
    and closing.
    """

    def __init__(self, file: h5py.File, group: str = "data", columns: Optional[Iterable[str]] = None, mask: Optional[np.ndarray] = None ):
        colnames = set(file["data"].keys())
        if columns is not None:
            colnames &= set(columns)

        if mask is not None:
            self.__data = {colname: file["data"][colname][mask] for colname in colnames}
        else:
            self.__data = {colname: file["data"][colname][()] for colname in colnames}

    def __len__(self) -> int:
        return len(next(iter(self.__data.values())))

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        return False

    def collect(self, columns: Iterable[str], mask: np.ndarray) -> InMemoryHandler:
        new_data = {colname: self.__data[colname][mask] for colname in columns}
        return InMemoryHandler(new_data)

    def write(
        self,
        file: h5py.File,
        mask: np.ndarray,
        columns: Iterable[str],
        dataset_name="data",
    ) -> None:
        group = file.require_group(dataset_name)
        for column in columns:
            group.create_dataset(column, data=self.__data[column][mask])

    def get_data(
        self,
        builders: dict = {},
        mask: Optional[np.ndarray] = None,
        n: Optional[int] = None,
        strategy: str = "start",
    ) -> Column | Table:
        """ """
        length = len(self)
        if n is not None and n > length:
            raise ValueError("Requested more data than is available.")

        n_to_take = n if n is not None else length
        sl: slice | np.ndarray
        if strategy == "start":
            sl = slice(0, n_to_take)
        elif strategy == "end":
            sl = slice(length - n_to_take, length)
        elif strategy == "random":
            idxs = np.random.choice(length, n_to_take, replace=False)
            sl = np.sort(idxs)

        data_mask = mask if mask is not None else np.ones(length, dtype=bool)

        output = {}
        for column, builder in builders.items():
            col = self.__data[column][data_mask][sl]
            output[column] = builder.build(Column(col, name=column))

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)

    def update_mask(self, n: int, strategy: str, mask: np.ndarray) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be greater than zero.")
        if n > np.sum(mask):
            raise ValueError("Requested more data than is available.")
        new_mask = np.zeros_like(mask)
        indices = np.where(mask)[0]
        if strategy == "start":
            new_mask[indices[:n]] = True
        elif strategy == "end":
            new_mask[indices[-n:]] = True
        elif strategy == "random":
            new_mask[np.random.choice(indices, n, replace=False)] = True
        else:
            raise ValueError(
                "Take strategy must be one of 'start', 'end', or 'random'."
            )
        return new_mask
