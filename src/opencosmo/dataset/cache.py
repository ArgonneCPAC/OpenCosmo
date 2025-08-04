from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from opencosmo.index import DataIndex


class ColumnCache:
    def __init__(self, columns: dict[str, NDArray]):
        self.__columns = columns

    @classmethod
    def empty(cls):
        return ColumnCache({})

    def with_columns(self, columns: Iterable[str]):
        new_columns = {
            key: self.__columns[key] for key in columns if key in self.__columns
        }
        return ColumnCache(new_columns)

    def keys(self):
        return self.__columns.keys()

    def add_column(self, name: str, column: np.ndarray):
        self.__columns[name] = column

    def with_mask(self, mask: DataIndex):
        new_columns = {name: mask.get_data(col) for name, col in self.__columns.items()}
        return ColumnCache(new_columns)

    def columns(self):
        yield from self.__columns.items()
