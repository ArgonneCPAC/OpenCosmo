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

    def __init__(self, file: h5py.File):
        colnames = file["data"].keys()
        self.__data = {colname: file["data"][colname][()] for colname in colnames}

    def __len__(self) -> int:
        return len(next(iter(self.__data.values())))

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        return False

    close = __exit__

    def write(
        self,
        file: h5py.File,
        filter: np.ndarray,
        columns: Iterable[str],
        dataset_name="data",
    ) -> None:
        group = file.require_group(dataset_name)
        for column in columns:
            group.create_dataset(column, data=self.__data[column][filter])

    def get_data(
        self,
        builders: dict = {},
        filter: Optional[np.ndarray] = None,
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

        data_filter = filter if filter is not None else np.ones(length, dtype=bool)

        output = {}
        for column, builder in builders.items():
            col = self.__data[column][data_filter][sl]
            output[column] = builder.build(Column(col, name=column))

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)

    def update_filter(self, n: int, strategy: str, filter: np.ndarray) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be greater than zero.")
        if n > np.sum(filter):
            raise ValueError("Requested more data than is available.")
        new_filter = np.zeros_like(filter)
        indices = np.where(filter)[0]
        if strategy == "start":
            new_filter[indices[:n]] = True
        elif strategy == "end":
            new_filter[indices[-n:]] = True
        elif strategy == "random":
            new_filter[np.random.choice(indices, n, replace=False)] = True
        else:
            raise ValueError(
                "Take strategy must be one of 'start', 'end', or 'random'."
            )
        return new_filter
