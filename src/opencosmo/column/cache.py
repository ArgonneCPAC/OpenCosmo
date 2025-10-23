from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional
from weakref import finalize, ref

from opencosmo.index.protocols import DataIndex
from opencosmo.index.take import take

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.index import DataIndex


def finish(
    cached_data: dict[str, np.ndarray],
    index: DataIndex,
    cache_ref: ref[ColumnCache],
):
    if not isinstance(cached_data, dict):
        raise ValueError(cached_data)
    cache = cache_ref()
    if cache is None:
        return

    data = {}
    for col in cache.columns:
        if col not in cached_data:
            continue
        data[col] = index.get_data(cached_data[col])
    if data:
        cache.add_data(data)


def check_length(cache: ColumnCache, data: dict[str, np.ndarray]):
    lengths = set(len(d) for d in data.values())
    if len(lengths) > 1:
        raise ValueError(
            "When adding data to the cache, all columns must be the same length"
        )
    elif (l := len(cache)) > 0 and l != lengths.pop():
        raise ValueError(
            "When adding data to the cache, the columns must be the same length as the columns currently in the cache"
        )


class ColumnCache:
    """
    A column cache is used to persist data that is read from an hdf5 file. Caches can get data in one of two ways:
    1. They are explicitly given data that has been recently read from disk or
    2. They take data from a previous cache

    ColumnCaches break some of the rules that most other things follow in this library, notably that they have internal
    state (which can change). This mutability is required for two reasons.

    1. If the parent cache is garbage collected, the child cache needs to be able to copy over any data it needs
    2. If a new cache is created by adding columns, we need to signal the child to update their parent to the new
       cache. This allows us to preserve the standard "operations create new objects" pattern that is present
       throughout the library.

    """

    def __init__(
        self,
        cached_data: dict[str, np.ndarray],
        columns: set[str],
        derived_index: Optional[DataIndex] = None,
        parent: Optional[ref[ColumnCache]] = None,
        children: list[ref[ColumnCache]] = [],
    ):
        self.__cached_data = cached_data
        self.__columns = columns
        self.__derived_index = derived_index
        self.__parent = parent
        self.__children = children
        self.__finalizer = None
        if parent is not None and (p := parent()) is not None:
            assert derived_index is not None
            self.__finalizer = finalize(
                p,
                finish,
                p.__cached_data,
                derived_index,
                ref(self),
            )

    @classmethod
    def empty(cls, columns: Iterable[str]):
        return ColumnCache({}, set(columns), None, None, [])

    @property
    def columns(self):
        return self.__columns

    def __update_parent(self, parent: ColumnCache):
        assert self.__parent is not None
        assert self.__derived_index is not None
        assert self.__finalizer is not None
        self.__finalizer.detach()
        self.__parent = ref(parent)
        self.__finalizer = finalize(
            parent, finish, parent.__cached_data, self.__derived_index, ref(self)
        )

    def __len__(self):
        if not self.__cached_data and self.__derived_index is None:
            return 0
        elif self.__derived_index is not None:
            return len(self.__derived_index)
        else:
            return len(next(iter(self.__cached_data.values())))

    def select(self, columns: Iterable[str]):
        columns = set(columns)
        if not self.__columns.issuperset(columns):
            raise ValueError("Tried to select columns that are not in this cache!")
        data = {
            key: self.__cached_data[key] for key in columns if key in self.__cached_data
        }
        return ColumnCache(data, columns, self.__derived_index, self.__parent)

    def add_data(self, data: dict[str, np.ndarray]):
        """
        The in-place equivalent of with_data. Should not be used outside the context of this
        file.

        """
        check_length(self, data)
        self.__cached_data = self.__cached_data | data

    def with_data(self, data: dict[str, np.ndarray]):
        check_length(self, data)
        new_cached_data = self.__cached_data | data
        new_cache = ColumnCache(
            new_cached_data,
            self.__columns,
            self.__derived_index,
            self.__parent,
            self.__children,
        )
        for child_ref in self.__children:
            if (child := child_ref()) is not None:
                child.__update_parent(new_cache)
        return new_cache

    def request(self, column_names: Iterable[str], index: DataIndex):
        column_names = set(column_names)
        columns_in_cache = column_names.intersection(self.__cached_data.keys())
        missing_columns = column_names - columns_in_cache

        data = {
            name: index.get_data(self.__cached_data[name]) for name in columns_in_cache
        }
        if self.__parent is None or column_names == columns_in_cache:
            return data

        parent = self.__parent()
        if parent is None:
            return data
        assert self.__derived_index is not None
        new_index = take(self.__derived_index, index)
        return data | parent.request(column_names, new_index)

    def take(self, index: DataIndex):
        if len(self) == 0:
            return ColumnCache.empty(self.__columns)
        if index.range()[1] > len(self):
            raise ValueError(
                "Tried to take more elements than the length of the cache!"
            )
        new_cache = ColumnCache({}, self.__columns, index, ref(self))
        self.__children.append(ref(new_cache))
        return new_cache

    def get_columns(self, columns: Iterable[str]):
        columns = set(columns)
        columns_in_cache = columns.intersection(self.__cached_data.keys())
        missing_columns = columns - columns_in_cache
        output = {c: self.__cached_data[c] for c in columns_in_cache}
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
        self.__cached_data = self.__cached_data | result
        return result
