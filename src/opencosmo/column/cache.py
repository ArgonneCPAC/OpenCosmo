from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional
from weakref import finalize, ref

from opencosmo.index.protocols import DataIndex
from opencosmo.index.take import take

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.index import DataIndex


def finish(
    columns: dict[str, np.ndarray], index: DataIndex, cache_ref: ref[ColumnCache]
):
    cache = cache_ref()
    if cache is None:
        return
    data = {name: index.get_data(col) for name, col in columns.items()}
    if data:
        cache.add_data(data)


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
        if parent is not None and (p := parent()) is not None:
            assert self.__derived_index is not None
            finalize(p, finish, p.__columns, self.__derived_index, ref(self))

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
            self.__parent = None
            self.__derived_index = None
            return False
        return parent.has(column_name)

    def request(self, column_names: Iterable[str], index: DataIndex):
        column_names = set(column_names)
        columns_in_cache = column_names.intersection(self.__columns.keys())
        missing_columns = column_names - columns_in_cache

        data = {name: index.get_data(self.__columns[name]) for name in columns_in_cache}
        if self.__parent is None or column_names == columns_in_cache:
            return data

        parent = self.__parent()
        if parent is None:
            self.__parent = None
            self.__derived_index = None
            return data
        assert self.__derived_index is not None
        new_index = take(self.__derived_index, index)
        return data | parent.request(column_names, new_index)

    def take(self, index: DataIndex):
        if index.range()[1] > len(self):
            raise ValueError(
                "Tried to take more elements than the length of the cache!"
            )
        return ColumnCache({}, index, ref(self))

    def get_columns(self, columns: Iterable[str]):
        columns = set(columns)
        columns_in_cache = columns.intersection(self.__columns.keys())
        missing_columns = columns - columns_in_cache
        output = {c: self.__columns[c] for c in columns_in_cache}
        output |= self.__get_derived_columns(missing_columns)
        return output

    def __get_derived_columns(self, column_names: set[str]):
        if self.__derived_index is None:
            return {}
        assert self.__parent is not None
        parent = self.__parent()
        if parent is None:
            return {}
        result = parent.request(column_names, self.__derived_index)
        self.__columns = self.__columns | result
        return result
