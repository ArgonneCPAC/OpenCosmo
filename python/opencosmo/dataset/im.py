from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from opencosmo.units.handler import UnitHandler


def resort(columns: dict[str, np.ndarray], sorted_index: Optional[np.ndarray]):
    if sorted_index is None or not columns:
        return columns
    reverse_sort = np.argsort(sorted_index)
    return {name: data[reverse_sort] for name, data in columns.items()}


def validate_in_memory_columns(
    columns: dict[str, np.ndarray], unit_handler: UnitHandler, ds_length: int
):
    new_units = {}
    for colname, column in columns.items():
        if len(column) != ds_length:
            raise ValueError(f"Column {colname} is not the same length as the dataset!")
        if isinstance(column, u.Quantity):
            new_units[colname] = column.unit
        else:
            new_units[colname] = None

    return unit_handler.with_static_columns(**new_units)
