from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import numpy as np
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import ColumnWriter

from opencosmo.index import get_data, get_length, into_array, take

if TYPE_CHECKING:
    from astropy.units import UnitBase
    from opencosmo.io.schema import Schema

    from opencosmo.index import DataIndex


@dataclass(frozen=True)
class InMemoryHandler:
    """Raw, baseline-convention arrays backing an in-memory dataset."""

    data: Mapping[str, np.ndarray]
    units: Mapping[str, UnitBase | None]
    column_descriptions: Mapping[str, str]
    _index: DataIndex

    def __len__(self) -> int:
        return get_length(self._index)

    def __derive(self, index: DataIndex) -> InMemoryHandler:
        return InMemoryHandler(self.data, self.units, self.column_descriptions, index)

    def with_index(self, index: DataIndex) -> InMemoryHandler:
        return self.__derive(index)

    def take(
        self, other: DataIndex, sorted: Optional[np.ndarray] = None
    ) -> InMemoryHandler:
        if sorted is None:
            return self.__derive(take(self._index, other))

        if get_length(sorted) != get_length(self._index):
            raise ValueError("Sorted index has the wrong length!")
        new_indices = get_data(other, sorted)
        return self.__derive(np.sort(into_array(self._index)[new_indices]))

    @property
    def index(self) -> DataIndex:
        return self._index

    @property
    def columns(self) -> list[str]:
        return list(self.data)

    @property
    def descriptions(self) -> Mapping[str, str]:
        return self.column_descriptions

    @property
    def load_conditions(self) -> None:
        return None

    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        return {name: get_data(self.data[name], self._index) for name in columns}

    def make_schema(self, columns: Iterable[str]) -> Schema:
        writers = {}
        for name in sorted(columns):
            unit = self.units[name]
            attrs = {
                "unit": "" if unit is None else str(unit),
                "description": self.column_descriptions.get(name, "None"),
            }
            writers[name] = ColumnWriter.from_numpy_array(
                get_data(self.data[name], self._index), attrs=attrs
            )
        if not writers:
            return make_schema("data", FileEntry.EMPTY)
        return make_schema("data", FileEntry.COLUMNS, columns=writers)
