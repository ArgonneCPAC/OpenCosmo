from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Optional

import astropy.units as u

from opencosmo.column.column import RawColumn
from opencosmo.io.schema import (
    FileEntry,
    combine_with_cached_schema,
    make_schema,
)
from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter, NumpySource

if TYPE_CHECKING:
    from opencosmo.column.column import ConstructedColumn
    from opencosmo.handler.protocols import DataCache, DataHandler
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region


def get_derived_column_names(
    producers: list[ConstructedColumn], columns: set[str]
) -> set[str]:
    all_derived: set[str] = reduce(
        lambda acc, col: acc.union(
            col.produces if not isinstance(col, RawColumn) else set()
        ),
        producers,
        set(),
    )
    return all_derived.intersection(columns)


def build_derived_writers(
    producers: list[ConstructedColumn],
    derived_data: dict,
    data_schema: Schema,
    cached_data_schema: Schema,
) -> None:
    """Add ColumnWriter entries to data_schema for each non-raw, non-cached producer."""
    for producer in producers:
        if isinstance(producer, RawColumn) or producer.produces.issubset(
            cached_data_schema.columns.keys()
        ):
            continue
        coldata = {name: derived_data[name] for name in producer.produces}
        units = {
            name: str(cd.unit) if isinstance(cd, u.Quantity) else ""
            for name, cd in coldata.items()
        }
        coldata = {
            name: cd.value if isinstance(cd, u.Quantity) else cd
            for name, cd in coldata.items()
        }
        for name, cd in coldata.items():
            attrs = {"unit": units[name], "description": producer.description}
            source = NumpySource(cd)
            writer = ColumnWriter([source], ColumnCombineStrategy.CONCAT, attrs=attrs)
            data_schema.columns[name] = writer


def make_dataset_schema(
    producers: list[ConstructedColumn],
    raw_data_handler: DataHandler,
    cache: DataCache,
    columns: set[str],
    meta_columns: list[str],
    header: OpenCosmoHeader,
    region: Region,
    derived_data: dict,
    name: Optional[str] = None,
) -> Schema:
    header = header.with_region(region)
    raw_columns = columns.intersection(raw_data_handler.columns)
    data_schema, metadata_schema = raw_data_handler.make_schema(raw_columns, header)

    cached_data_schema, cached_metadata_schema = cache.make_schema(
        list(columns) + meta_columns
    )

    build_derived_writers(producers, derived_data, data_schema, cached_data_schema)

    attributes = {}
    if (load_conditions := raw_data_handler.load_conditions) is not None:
        attributes["load/if"] = load_conditions

    data_schema = combine_with_cached_schema(data_schema, cached_data_schema)
    metadata_schema = combine_with_cached_schema(
        metadata_schema, cached_metadata_schema
    )

    children = {"data": data_schema}
    if metadata_schema.type != FileEntry.EMPTY:
        children[metadata_schema.name] = metadata_schema
    if name is None:
        name = ""

    return make_schema(
        name, FileEntry.DATASET, children=children, attributes=attributes
    )
