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


def combine_with_cached_schema(raw_data_schema, cached_schema):
    if raw_data_schema is None or raw_data_schema.type == FileEntry.EMPTY:
        return cached_schema
    elif cached_schema is None or cached_schema.type == FileEntry.EMPTY:
        return raw_data_schema

    all_column_names = set(raw_data_schema.columns.keys()).union(
        cached_schema.columns.keys()
    )

    new_columns = {}
    for column in all_column_names:
        new_attrs = {}
        if column in raw_data_schema.columns:
            new_columns[column] = raw_data_schema.columns[column]
            new_attrs = raw_data_schema.columns[column].attrs
        if column in cached_schema.columns:
            new_columns[column] = cached_schema.columns[column]
            new_columns[column].update_attrs(new_attrs)

    return make_schema(
        raw_data_schema.name,
        type_=FileEntry.COLUMNS,
        columns=new_columns,
        attributes=raw_data_schema.attributes,
    )
