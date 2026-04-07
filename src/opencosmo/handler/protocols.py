from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Protocol, Self

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema


class DataHandler(Protocol):
    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        """ """

    def get_metadata(self, columns: Iterable[str]) -> dict[str, np.ndarray]: ...

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None) -> Self: ...

    def make_schema(
        self, columns: Iterable[str], header: Optional[OpenCosmoHeader] = None
    ) -> tuple[Schema, Schema]: ...

    @property
    def columns(self) -> Iterable[str]: ...
    @property
    def metadata_columns(self) -> Iterable[str]: ...
    @property
    def load_conditions(self) -> Optional[dict]: ...

    @property
    def index(self) -> DataIndex: ...


class DataCache(DataHandler, Protocol):
    def add_data(
        self,
        data: dict[str, np.ndarray],
        descriptions: dict[str, str],
        push_up: bool = True,
    ): ...

    def __len__(self) -> int: ...

    def drop(self, columns: Iterable[str]) -> Self: ...
    def register_column_group(self, state_id: int, columns: set[str]) -> None: ...
    def deregister_column_group(self, state_id: int) -> None: ...
    def create_child(self) -> Self: ...
