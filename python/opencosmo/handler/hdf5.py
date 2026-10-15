from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Optional

import h5py
import numpy as np
from opencosmo.index import (
    SimpleIndex,
    get_data,
    get_length,
    into_array,
    read_columns,
    take,
)
from opencosmo.index.direct import chunk_map
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import (
    ColumnWriter,
)

if TYPE_CHECKING:
    from opencosmo.index import DataIndex
    from opencosmo.index.direct import ChunkMap, LastChunks
    from opencosmo.io.schema import Schema


class Hdf5Handler:
    """
    Handler for opencosmo.Dataset

    Holds the live ``/data`` group rather than pre-opened column datasets. A column's
    ``h5py.Dataset`` is acquired the first time it is actually read, which keeps an
    open proportional to the columns a caller touches instead of to the columns a file
    contains. Good for network file systems.

    ``column_names`` comes from the discovered ``GroupLayout``, which also guarantees
    every column shares ``row_count``.

    Columns stored with Blosc2 are read straight from the file (see
    ``opencosmo.index.direct``). Their chunk maps and last decoded chunks are kept
    per data group and shared with derived handlers, like the column handles.
    """

    def __init__(
        self,
        data_group: h5py.Group,
        columns: dict[str, h5py.Dataset | None],
        index: DataIndex,
        load_conditions: Optional[dict[str, bool]] = None,
        descriptions: Optional[dict[str, str | None]] = None,
        chunk_maps: Optional[dict[str, ChunkMap | None]] = None,
        last_chunks: Optional[LastChunks] = None,
    ):
        self.__data_group = data_group
        self.__columns = columns
        self.__chunk_maps = chunk_maps if chunk_maps is not None else {}
        self.__last_chunks = last_chunks if last_chunks is not None else {}
        self.__index = index
        self.__load_conditions = load_conditions
        self.__descriptions = descriptions

    def __handle(self, name: str) -> h5py.Dataset:
        handle = self.__columns.get(name)
        if handle is None:
            handle = _open_column(self.__data_group, name)
            self.__columns[name] = handle
            # Modified in-place so downstream handlers share.
        return handle

    def __source(self, name: str) -> h5py.Dataset | ChunkMap:
        if name not in self.__chunk_maps:
            self.__chunk_maps[name] = chunk_map(self.__handle(name))
        return self.__chunk_maps[name] or self.__handle(name)

    def __len__(self):
        return get_length(self.__index)

    def with_index(self, index: DataIndex) -> Hdf5Handler:
        return self.__derive(index)

    def __derive(self, index: DataIndex) -> Hdf5Handler:
        return Hdf5Handler(
            self.__data_group,
            self.__columns,
            index,
            self.__load_conditions,
            descriptions=self.__descriptions,
            chunk_maps=self.__chunk_maps,
            last_chunks=self.__last_chunks,
        )

    @property
    def load_conditions(self) -> Optional[dict[str, bool]]:
        return self.__load_conditions

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None):
        if len(other) == 0:
            return self.__derive(other)

        if sorted is not None:
            return self.__take_sorted(other, sorted)

        return self.__derive(take(self.__index, other))

    def __take_sorted(self, other: DataIndex, sorted: np.ndarray):
        if get_length(sorted) != get_length(self.__index):
            raise ValueError("Sorted index has the wrong length!")
        new_indices = get_data(other, sorted)

        new_raw_index = into_array(self.__index)[new_indices]
        new_index = np.sort(new_raw_index)

        return self.__derive(new_index)

    @property
    def index(self):
        return self.__index

    @cached_property
    def columns(self):
        return list(self.__columns.keys())

    @property
    def descriptions(self):
        if self.__descriptions is not None:
            return self.__descriptions
        return {
            name: self.__handle(name).attrs.get("description")
            for name in self.__columns
        }

    def mask(self, mask):
        idx = SimpleIndex(np.where(mask)[0])
        return self.take(idx)

    def make_schema(
        self,
        columns: Iterable[str],
    ) -> Schema:
        columns = set(columns)
        data_writers = {}
        for column_name in columns:
            column = self.__handle(column_name)
            data_writers[column_name] = ColumnWriter.from_h5_dataset(
                column, self.__index, attrs=dict(column.attrs)
            )
        data_schema = make_schema("data", FileEntry.COLUMNS, columns=data_writers)

        return data_schema

    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        """ """
        # Ensure order is preserved
        return read_columns(
            {name: self.__source(name) for name in columns},
            self.__index,
            self.__last_chunks,
        )

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]


def _open_column(data_group: h5py.Group, name: str) -> h5py.Dataset:
    """
    Open a column with a chunk cache that holds at least one of its chunks.
    Successive small reads, such as iterating over structures, then decompress
    each chunk once instead of once per read.
    """
    column = data_group[name]
    if column.chunks is None:
        return column
    chunk_bytes = column.dtype.itemsize * int(np.prod(column.chunks))
    n_slots, n_bytes, w0 = column.id.get_access_plist().get_chunk_cache()
    if chunk_bytes <= n_bytes:
        return column
    # HDF5 applies cache settings only when a dataset is first opened, so the
    # probing handle must be released before reopening.
    del column
    access = h5py.h5p.create(h5py.h5p.DATASET_ACCESS)
    access.set_chunk_cache(n_slots, chunk_bytes, w0)
    return h5py.Dataset(h5py.h5d.open(data_group.id, name.encode(), access))
