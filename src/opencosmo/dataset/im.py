from typing import Iterable

import astropy.units as u  # type: ignore
import numpy as np
from numpy.typing import NDArray

from opencosmo.index import DataIndex


class InMemoryColumnHandler:
    def __init__(
        self,
        columns: dict[str, NDArray | u.Quantity],
        index: DataIndex,
    ):
        self.__index = index
        self.__columns = columns

    @classmethod
    def empty(cls, index: DataIndex):
        return InMemoryColumnHandler({}, index)

    def with_columns(self, columns: Iterable[str]):
        new_columns = {
            key: self.__columns[key] for key in columns if key in self.__columns
        }
        return InMemoryColumnHandler(new_columns, self.__index)

    def keys(self):
        return self.__columns.keys()

    def with_new_column(self, name: str, column: np.ndarray | u.Quantity):
        if len(column) != len(self.__index):
            raise ValueError("Tried to add an in-memory column with the wrong length!")
        new_columns = {**self.__columns, name: column}
        return InMemoryColumnHandler(new_columns, self.__index)

    def project(self, index: DataIndex):
        if len(self.__columns) == 0:
            return InMemoryColumnHandler.empty(index)
        index_into_columns = self.__index.projection(index)

        new_columns = {
            name: index_into_columns.get_data(col)
            for name, col in self.__columns.items()
        }
        return InMemoryColumnHandler(new_columns, index)

    def columns(self):
        yield from self.__columns.items()
