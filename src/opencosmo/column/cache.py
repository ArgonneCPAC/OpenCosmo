from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Optional
from weakref import finalize, ref

import astropy.units as u
import numpy as np

from opencosmo.index import DataIndex
from opencosmo.index.get import get_data
from opencosmo.index.take import take
from opencosmo.index.unary import get_length, get_range

if TYPE_CHECKING:
    from opencosmo.index import DataIndex


ColumnUpdater = Callable[[np.ndarray | u.Quantity], np.ndarray | u.Quantity]


def finish(
    cached_data: dict[str, np.ndarray],
    index: Optional[DataIndex],
    cache_ref: ref[ColumnCache],
):
    cache = cache_ref()
    if cache is None:
        return

    columns_to_add = (
        cache.registered_columns.intersection(cached_data.keys()) - cache.columns
    )
    data = {name: cached_data[name] for name in columns_to_add}
    if index is not None:
        data = {name: get_data(cd, index) for name, cd in data.items()}
    if data:
        cache.add_data(data)


def check_length(cache: ColumnCache, data: dict[str, np.ndarray]):
    lengths = set(len(d) for d in data.values())
    if len(lengths) > 1:
        raise ValueError(
            "When adding data to the cache, all columns must be the same length"
        )
    elif (length := len(cache)) > 0 and length != lengths.pop():
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
        registered_column_groups: dict[int, set[str]],
        column_descriptions: dict[str, str],
        derived_index: Optional[DataIndex],
        parent: Optional[ref[ColumnCache]],
        children: Optional[list[ref[ColumnCache]]],
    ):
        self.__cached_data = cached_data
        self.__registered_column_groups = registered_column_groups
        self.__descriptions = column_descriptions
        self.__derived_index = derived_index
        self.__parent = parent
        if children is None:
            children = []
        self.__children = children
        self.__finalizer = None

        if parent is not None and (p := parent()) is not None:
            self.__finalizer = finalize(
                p,
                finish,
                p.__cached_data,
                derived_index,
                ref(self),
            )
            self.__finalizer.atexit = False  # type: ignore

    @classmethod
    def empty(cls):
        return ColumnCache({}, {}, {}, None, None, [])

    @property
    def columns(self):
        return set(self.__cached_data.keys())

    @property
    def descriptions(self):
        return self.__descriptions

    @property
    def registered_columns(self):
        return set().union(*list(self.__registered_column_groups.values()))

    def duplicate(self):
        return ColumnCache({}, {}, {}, None, ref(self), [])

    def __push_down(self, data: dict[str, np.ndarray]):
        columns_to_keep = self.registered_columns.intersection(data.keys()).difference(
            self.__cached_data.keys()
        )
        cached_data = {colname: data[colname] for colname in columns_to_keep}
        if self.__derived_index is not None:
            cached_data = {
                colname: get_data(coldata, self.__derived_index)
                for colname, coldata in cached_data.items()
            }

        self.__cached_data |= cached_data

    def __push_up(self, data: dict[str, np.ndarray]):
        assert len(self) == 0 or all(len(d) == len(self) for d in data.values())
        columns_to_keep = self.registered_columns.intersection(data.keys()).difference(
            self.__cached_data.keys()
        )
        self.__cached_data |= {key: data[key] for key in columns_to_keep}

    def register_column_group(self, key: int, data: set[str]):
        assert key not in self.__registered_column_groups
        self.__registered_column_groups[key] = data

    def deregister_column_group(self, state_id: int):
        assert state_id in self.__registered_column_groups
        columns = self.__registered_column_groups.pop(state_id)
        remaining_columns = set().union(*list(self.__registered_column_groups.values()))

        to_drop = columns.difference(remaining_columns)
        cached_data = {
            name: self.__cached_data.pop(name)
            for name in to_drop
            if name in self.__cached_data
        }
        if not cached_data:
            return
        for child_ in self.__children:
            if (child := child_()) is None:
                continue
            child.__push_down(cached_data)

    def __update_parent(self, parent: ColumnCache):
        assert self.__parent is not None
        assert self.__finalizer is not None
        self.__finalizer.detach()
        self.__parent = ref(parent)
        self.__finalizer = finalize(
            parent, finish, parent.__cached_data, self.__derived_index, ref(self)
        )
        self.__finalizer.atexit = False  # type: ignore

    def __len__(self):
        if not self.__cached_data and self.__derived_index is None:
            return 0
        elif self.__derived_index is not None:
            return get_length(self.__derived_index)
        elif self.__cached_data:
            return len(next(iter(self.__cached_data.values())))
        elif self.__parent is not None and (p := self.__parent()) is not None:
            return len(p)
        return 0

    def add_data(
        self,
        data: dict[str, np.ndarray],
        descriptions: dict[str, str] = {},
        push_up=True,
    ):
        """
        The in-place equivalent of with_data. Should not be used outside the context of this
        file.

        """
        check_length(self, data)

        self.__descriptions |= descriptions
        if (
            push_up
            and self.__derived_index is None
            and self.__parent is not None
            and (p := self.__parent()) is not None
        ):
            p.__push_up(data)

        self.__cached_data = self.__cached_data | data

    def with_data(
        self,
        data: dict[str, np.ndarray],
        descriptions: dict[str, str] = {},
        no_push_up=True,
    ):
        check_length(self, data)
        new_cached_data = self.__cached_data | data
        new_cache = ColumnCache(
            new_cached_data,
            self.__registered_column_groups,
            self.__descriptions | descriptions,
            self.__derived_index,
            self.__parent,
            self.__children,
        )
        for child_ref in self.__children:
            if (child := child_ref()) is not None:
                child.__update_parent(new_cache)

        return new_cache

    def without_columns(self, columns: Iterable[str]):
        columns_to_drop = set(self.__cached_data.keys()).intersection(columns)
        data = {
            name: data
            for name, data in self.__cached_data.items()
            if name not in columns_to_drop
        }
        descriptions = {
            name: desc
            for name, desc in self.__descriptions.items()
            if name not in columns_to_drop
        }
        return ColumnCache(data, {}, descriptions, None, None, [])

    def drop(self, columns: Iterable[str]):
        columns_in_cache = set(self.__cached_data.keys()).intersection(columns)
        for column in columns_in_cache:
            del self.__cached_data[column]

    def request(self, column_names: Iterable[str], index: Optional[DataIndex]):
        column_names = set(column_names)
        columns_in_cache = column_names.intersection(self.__cached_data.keys())

        data = {name: self.__cached_data[name] for name in columns_in_cache}
        if index is not None:
            data = {name: get_data(cd, index) for name, cd in data.items()}

        if self.__parent is None or column_names == columns_in_cache:
            return data

        parent = self.__parent()
        if parent is None:
            return data

        match (index, self.__derived_index):
            case (None, None):
                new_index = None
            case (_, None):
                new_index = index
            case (None, _):
                new_index = self.__derived_index
            case _:
                assert self.__derived_index is not None and index is not None
                new_index = take(self.__derived_index, index)

        return data | parent.request(column_names, new_index)

    def take(self, index: DataIndex):
        if len(self) == 0:
            return ColumnCache.empty()
        if get_range(index)[1] > len(self):
            raise ValueError(
                "Tried to take more elements than the length of the cache!"
            )
        new_cache = ColumnCache({}, {}, {}, index, ref(self), [])
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
        if self.__parent is None:
            return {}
        parent = self.__parent()
        if parent is None:
            return {}
        result = parent.request(column_names, self.__derived_index)

        self.__cached_data = self.__cached_data | result
        return result
