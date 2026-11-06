"""Live parameter types for recursive structure collection transformations."""

from collections.abc import Iterable
from typing import TypedDict

import astropy.units as u
import numpy as np

from opencosmo.column.column import ConstructedColumn

type ColumnNames = str | Iterable[str]
type SelectionColumn = ConstructedColumn | np.ndarray | u.Quantity


class DerivedSelection(TypedDict, total=False):
    """Explicit columns and expressions selected from one dataset."""

    columns: Iterable[str]
    derived_columns: dict[str, SelectionColumn]


type StructureSelection = (
    ColumnNames | SelectionColumn | DerivedSelection | dict[str, StructureSelection]
)
type StructureDrop = ColumnNames | dict[str, StructureDrop]
type UnitConversions = dict[u.UnitBase, u.UnitBase]
type StructureUnitSettings = dict[
    str, u.UnitBase | UnitConversions | StructureUnitSettings
]

__all__ = [
    "ColumnNames",
    "DerivedSelection",
    "SelectionColumn",
    "StructureDrop",
    "StructureSelection",
    "StructureUnitSettings",
    "UnitConversions",
]
