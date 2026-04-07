from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Self

from opencosmo.index import empty
from opencosmo.io.schema import FileEntry, make_schema

if TYPE_CHECKING:
    import numpy as np
    from opencosmo.index import DataIndex


class EmptyHandler:
    def get_data(self, *args):
        return {}

    def get_metadata(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        return {}

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None) -> Self:
        return self

    def __len__(self) -> int:
        return 0

    def make_schema(self, *args, **kwargs):
        data_schema = make_schema("data", FileEntry.EMPTY)
        metadata_schema = make_schema("metadata", FileEntry.EMPTY)
        return data_schema, metadata_schema

    @property
    def columns(self) -> Iterable[str]:
        return set()

    @property
    def load_conditions(self):
        return None

    @property
    def metadata_columns(self) -> Iterable[str]:
        return set()

    @property
    def descriptions(self):
        return {}

    @property
    def index(self) -> DataIndex:
        return empty()
