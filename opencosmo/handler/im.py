from copy import copy
from typing import Optional

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

    def get_data(
        self, builders: dict = {}, filter: Optional[np.ndarray] = None
    ) -> Column | Table:
        """ """
        output = {}
        for column, builder in builders.items():
            if filter is None:
                output[column] = Column(copy(self.__data[column]))
            else:
                output[column] = Column(copy(self.__data[column][filter]))

            output[column] = builder.build(output[column])

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)
