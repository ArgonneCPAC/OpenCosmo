from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

if TYPE_CHECKING:
    from .writer import ColumnWriter


class FileEntry(Enum):
    DATASET = "dataset"
    MULTI_DATASET = "multi_dataset"
    STRUCTURE_COLLECTION = "structure_collection"
    SIMULATION_COLLECTION = "simulation_collection"
    LIGHTCONE = "lightcone"
    LIGHCONE_MAP = "lightcone_map"
    HEALPIX_MAP = "healpix_map"
    COLUMNS = "columns"
    METADATA = "metadata"
    EMPTY = "empty"


class Schema(NamedTuple):
    name: str
    type: FileEntry
    children: dict[str, Schema]
    columns: dict[str, ColumnWriter]
    attributes: dict[str, Any]


def empty_schema(name: str, type_: FileEntry) -> Schema:
    return Schema(name, type_, {}, {}, {})


def make_schema(
    name: str,
    type_: FileEntry,
    children: Optional[dict] = None,
    columns: Optional[dict] = None,
    attributes: Optional[dict] = None,
):
    if children is None:
        children = {}
    if columns is None:
        columns = {}
    if attributes is None:
        attributes = {}
    return Schema(name, type_, children, columns, attributes)
