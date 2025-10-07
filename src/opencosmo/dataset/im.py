from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

if TYPE_CHECKING:
    import astropy.units as u
    from numpy.typing import NDArray

    from opencosmo.index import DataIndex


class InMemoryColumnHandler:
    def __init__(
        self,
        columns: dict[str, NDArray | u.Quantity],
        descriptions: dict[str, Optional[str]] = {},
    ):
        self.__columns = columns
        self.__descriptions = descriptions
        lengths = set(len(c) for c in columns.values())
        if len(lengths) > 1:
            raise ValueError(
                "Tried to instantiate an InMemoryColumnHandler with columns of different lengths!"
            )
        elif len(lengths) == 0:
            lengths = set([0])
        self.__length = lengths.pop()

    @property
    def descriptions(self):
        return self.__descriptions

    @classmethod
    def empty(cls, index: DataIndex):
        return InMemoryColumnHandler({}, {})

    def __len__(self):
        return self.__length

    def with_columns(self, columns: Iterable[str]):
        new_columns = {
            key: self.__columns[key] for key in columns if key in self.__columns
        }
        return InMemoryColumnHandler(new_columns, self.__descriptions)

    def get_rows(self, index: np.ndarray):
        if self.__length == 0:
            return InMemoryColumnHandler({}, {})

        if index.dtype == np.bool_ and len(index) > self.__length:
            raise ValueError("Received a boolean index that is longer than the data!")

        if np.max(index) > self.__length:
            raise ValueError("Tried to get a rows that are not in this column!")

        new_data = {key: data[index] for key, data in self.__columns.items()}
        return InMemoryColumnHandler(new_data, self.__descriptions)

    def take_range(self, start: int, end: int):
        if self.__length == 0:
            return self
        elif start >= self.__length or end > self.__length:
            raise ValueError(
                "The requested range is outside the range of this dataset!"
            )
        new_data = {key: data[start:end] for key, data in self.__columns.items()}
        return InMemoryColumnHandler(new_data, self.__descriptions)

    def take_index(self, index: DataIndex):
        if len(self) == 0:
            return self
        new_data = {key: index.get_data(value) for key, value in self.__columns.items()}
        return InMemoryColumnHandler(new_data, self.__descriptions)

    def keys(self):
        return self.__columns.keys()

    def get_data(self, keys: str | Iterable[str]):
        if isinstance(keys, str):
            keys = [keys]
        missing = set(keys) - set(self.keys())
        if missing:
            raise ValueError(f"Requested unknown columns {missing}")
        return {name: self.__columns[name] for name in keys}

    def with_new_column(
        self,
        name: str,
        column: np.ndarray | u.Quantity,
        description: Optional[str] = None,
    ):
        if self.__columns and len(column) != len(self):
            raise ValueError("Tried to add an in-memory column with the wrong length!")
        new_columns = {**self.__columns, name: column}
        new_descriptions = {**self.__descriptions, name: description}
        return InMemoryColumnHandler(new_columns, new_descriptions)

    def columns(self):
        yield from self.__columns.items()
