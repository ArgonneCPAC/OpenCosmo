from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional
from weakref import WeakValueDictionary, ref

from opencosmo.index.protocols import DataIndex
from opencosmo.index.take import take

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.index import DataIndex


class ColumnCache:
    """
    A column cache is used to persist data that is read from an hdf5 file. Caches can get data in one of two ways:
    1. They are explicitly given data that has been recently read from disk or
    2. They take data from a previous cache


    """

    def __init__(
        self,
        columns: dict[str, np.ndarray],
        derived_index: Optional[DataIndex] = None,
        parent: Optional[ref[ColumnCache]] = None,
    ):
        self.__columns = columns
        self.__derived_index = derived_index
        self.__parent = parent

    @classmethod
    def empty(cls):
        return ColumnCache({})

    def __len__(self):
        if not self.__columns and self.__derived_index is None:
            return 0
        elif self.__derived_index is not None:
            return len(self.__derived_index)
        else:
            return len(next(iter(self.__columns.values())))

    def add_data(self, data: dict[str, np.ndarray]):
        lengths = set(len(d) for d in data.values())
        if len(lengths) > 1:
            raise ValueError(
                "When adding data to the cache, all columns must be the same length"
            )
        elif (l := len(self)) > 0 and l != lengths.pop():
            raise ValueError(
                "When adding data to the cache, the columns must be the same length as the columns currently in the cache"
            )

        self.__columns = self.__columns | data

    def has(self, column_name):
        if column_name in self.__columns:
            return True
        if self.__parent is None:
            return False
        parent = self.__parent()
        if parent is None:
            return False
        return parent.has(column_name)

    def request(self, column_name: str, index: DataIndex):
        if column_name in self.__columns:
            return index.get_data(self.__columns[column_name])
        elif self.__parent is None:
            return None
        parent = self.__parent()
        if parent is None or not parent.has(column_name):
            return None
        assert self.__derived_index is not None
        new_index = take(self.__derived_index, index)
        return parent.take(new_index)

    def take(self, index: DataIndex):
        if index.range()[1] > len(self):
            raise ValueError(
                "Tried to take more elements than the length of the cache!"
            )
        return ColumnCache({}, index, ref(self))

    def get_columns(self, columns: Iterable[str]):
        columns = set(columns)
        output = {}
        for column in columns:
            if (existing_column := self.__columns.get(column)) is not None:
                output[column] = existing_column
            elif (derived_column := self.__get_derived_column(column)) is not None:
                output[column] = derived_column
        return output

    def __get_derived_column(self, column_name: str):
        if self.__derived_index is None:
            return None
        assert self.__parent is not None
        parent = self.__parent()
        if parent is None:
            return None
        result = parent.request(column_name, self.__derived_index)
        if result is not None:
            self.__columns[column_name] = result
        return result
