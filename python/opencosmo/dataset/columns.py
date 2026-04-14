from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Optional

import astropy.units as u
import numpy as np

from opencosmo.column.column import Column, DerivedColumn, EvaluatedColumn, RawColumn
from opencosmo.dataset.graph import validate_column_producers

if TYPE_CHECKING:
    from opencosmo.column.column import ConstructedColumn
    from opencosmo.handler.protocols import DataCache
    from opencosmo.units.handler import UnitHandler


def resort(columns: dict[str, np.ndarray], sorted_index: Optional[np.ndarray]):
    if sorted_index is None or not columns:
        return columns
    reverse_sort = np.argsort(sorted_index)
    return {name: data[reverse_sort] for name, data in columns.items()}


def validate_in_memory_columns(
    columns: dict[str, np.ndarray], unit_handler: UnitHandler, ds_length: int
) -> UnitHandler:
    new_units = {}
    for colname, column in columns.items():
        if len(column) != ds_length:
            raise ValueError(f"Column {colname} is not the same length as the dataset!")
        if isinstance(column, u.Quantity):
            new_units[colname] = column.unit
        else:
            new_units[colname] = None

    return unit_handler.with_static_columns(**new_units)


def __categorize_columns(
    new_columns: dict,
    descriptions: dict[str, str],
    ds_length: int,
) -> tuple[list[ConstructedColumn], dict, dict, list[str], dict]:
    """
    Classify incoming columns by type and build the producer list, in-memory column
    dicts, and static unit map needed by add_columns.

    Returns:
        new_derived_columns, new_in_memory_columns, new_in_memory_descriptions,
        new_column_names, new_static_units
    """
    new_derived_columns: list[ConstructedColumn] = []
    new_in_memory_columns: dict = {}
    new_in_memory_descriptions: dict = {}
    new_column_names: list[str] = []
    new_static_units: dict = {}

    for colname, column in new_columns.items():
        match column:
            case DerivedColumn():
                column.name = colname
                column.description = descriptions.get(colname, "None")
                new_derived_columns.append(column)
                new_column_names.extend(column.produces)
            case EvaluatedColumn():
                column.description = descriptions.get(colname, "None")
                new_derived_columns.append(column)
                new_column_names.extend(column.produces)
            case Column():
                producer = RawColumn(
                    column.name, descriptions.get(colname, None), alias=colname
                )
                new_derived_columns.append(producer)
                new_column_names.extend(producer.produces)
            case np.ndarray():
                if len(column) != ds_length:
                    raise ValueError(
                        f"New column {colname} does not have the same length as this dataset!"
                    )
                new_in_memory_descriptions[colname] = descriptions.get(colname, "None")
                new_in_memory_columns[colname] = column
                new_column_names.append(colname)
                new_derived_columns.append(RawColumn(colname, None))
                new_static_units[colname] = (
                    column.unit if isinstance(column, u.Quantity) else None
                )
            case _:
                raise ValueError(f"Got an invalid new column of type {type(column)}")

    return (
        new_derived_columns,
        new_in_memory_columns,
        new_in_memory_descriptions,
        new_column_names,
        new_static_units,
    )


def add_columns(
    producers: list[ConstructedColumn],
    unit_handler: UnitHandler,
    cache: DataCache,
    existing_column_names: list[str],
    sorted_index: np.ndarray | None,
    descriptions: dict[str, str],
    new_columns: dict,
    ds_length: int,
) -> tuple[list[ConstructedColumn], list[str], UnitHandler]:
    existing_columns = set(existing_column_names)
    if inter := existing_columns.intersection(new_columns.keys()):
        raise ValueError(f"Some columns are already in the dataset: {inter}")

    (
        new_derived_columns,
        new_in_memory_columns,
        new_in_memory_descriptions,
        new_column_names,
        new_static_units,
    ) = __categorize_columns(new_columns, descriptions, ds_length)

    new_unit_handler = unit_handler.with_static_columns(**new_static_units)
    new_producers = copy(producers) + new_derived_columns

    new_units = validate_column_producers(new_producers, new_unit_handler)
    if new_units:
        new_unit_handler = new_unit_handler.with_new_columns(**new_units)

    if new_in_memory_columns:
        new_unit_handler = validate_in_memory_columns(
            new_in_memory_columns, unit_handler, ds_length
        )
        new_in_memory_columns = resort(new_in_memory_columns, sorted_index)
        cache.add_data(new_in_memory_columns, descriptions=new_in_memory_descriptions)

    return new_producers, existing_column_names + new_column_names, new_unit_handler
