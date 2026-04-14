from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Protocol, Self

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.schema import Schema

    from opencosmo.index import DataIndex


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


class DataCache(Protocol):
    def add_data(
        self,
        data: dict[UUID, dict[str, np.ndarray]],
        descriptions: dict[str, str],
        push_up: bool = True,
    ): ...

    def add_metadata(
        self,
        data: dict[str, np.ndarray],
        descriptions: dict[str, str] = {},
    ): ...

    def get_data(
        self, pairs: set[tuple[UUID, str]]
    ) -> dict[UUID, dict[str, np.ndarray]]: ...

    def get_metadata(self, column_names: Iterable[str]) -> dict[str, np.ndarray]: ...

    def __len__(self) -> int: ...

    def take(self, index: DataIndex) -> Self: ...

    def drop(self, columns: Iterable[str]) -> Self: ...

    def register_column_group(
        self, state_id: int, columns: dict[str, UUID]
    ) -> None: ...

    def deregister_column_group(self, state_id: int) -> None: ...

    def create_child(self) -> Self: ...

    @property
    def columns(self) -> set[str]: ...

    @property
    def metadata_columns(self) -> set[str]: ...

    @property
    def descriptions(self) -> dict[str, str]: ...

    def make_schema(
        self, columns: dict[str, UUID], meta_columns: list[str]
    ) -> tuple[Schema, Schema]: ...
