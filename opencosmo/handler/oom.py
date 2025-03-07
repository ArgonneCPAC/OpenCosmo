from copy import copy
from typing import Iterable, Optional

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore


class OutOfMemoryHandler:
    """
    A handler for in-memory storage. Data will remain on disk until needed

    """

    def __init__(self, file: h5py.File, group: str = "data"):
        self.__file = file
        self.__group = file[group]
        self.__columns = list(self.__group.keys())


    def __len__(self) -> int:
        return self.__group[self.__columns[0]].size

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        self.__columns = None
        return self.__file.close()

    def write(
        self,
        file: h5py.File,
        filter: np.ndarray,
        columns: Iterable[str],
        dataset_name="data",
    ) -> None:
        group = file.require_group(dataset_name)
        for column in columns:
            data = self.__group[column][filter]
            group.create_dataset(column, data=data)

    def get_data(
        self, builders: dict = {}, filter: Optional[np.ndarray] = None
    ) -> Column | Table:
        """ """
        if self.__group is None:
            raise ValueError("This file has already been closed")
        output = {}
        for column, builder in builders.items():
            if filter is None:
                data = self.__group[column][()]
            else:
                data = self.__group[column][filter]

            col = Column(data, name=column)
            output[column] = builder.build(col)

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)
