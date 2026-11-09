"""The methods exposed as messages and how their parameters travel over the wire.

Everything that cannot be inferred from a method signature lives here: the
methods each target exposes, the wire type of each live type, and semantic
checks that span several parameters. Edit this file, then regenerate with
``python -m opencosmo.serde.codegen``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import NoneType

import astropy.units as u
import numpy as np
import numpy.typing as npt
from annotated_types import Ge

from opencosmo.collection.lightcone import Lightcone
from opencosmo.collection.structure import StructureCollection
from opencosmo.collection.structure import types as sct
from opencosmo.column.column import (
    ColumnMask,
    CompoundColumnMask,
    ConstructedColumn,
    DerivedScalarValue,
)
from opencosmo.dataset import Dataset
from opencosmo.index import DataIndex
from opencosmo.spatial.protocols import Region

from .. import checks, live
from .. import structure as scwire
from ..common import RowIndices, UnitString
from ..expression import Expression, FiniteNumber, Mask
from ..region import RegionMessage


@dataclass(frozen=True)
class Symbol:
    """A name importable from a module."""

    module: str
    name: str


def symbol(value: object) -> Symbol:
    """Find where a class, function, or type alias can be imported from.

    Raises
    ------
    TypeError
        If ``value`` is not importable under its own module and name.
    """
    module = getattr(value, "__module__", None)
    name = getattr(value, "__name__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        raise TypeError(f"{value!r} has no importable name")
    if module != "builtins":
        import importlib

        if getattr(importlib.import_module(module), name, None) is not value:
            raise TypeError(f"{value!r} cannot be imported as {module}.{name}")
    return Symbol(module, name)


@dataclass(frozen=True)
class WireType:
    """The named wire annotation for a live annotation.

    ``to_live`` converts one validated wire value into the live value the method
    expects. It is ``None`` when the wire value can be passed through unchanged.
    """

    annotation: Symbol
    to_live: Symbol | None = None


@dataclass(frozen=True)
class Check:
    """A semantic check called with the values of the named message fields."""

    function: Symbol
    fields: tuple[str, ...]


@dataclass(frozen=True)
class MessageSpec:
    """The messages generated for one target type.

    ``namespace`` supplies names the target's module imports only for type
    checking, so its annotations can be resolved.
    """

    target: type
    public_name: str
    methods: tuple[str, ...]
    output: Path
    namespace: Mapping[str, object]
    checks: Mapping[str, tuple[Check, ...]] = field(default_factory=dict)


def __wire(
    annotation: object, to_live: Callable[..., object] | None = None
) -> WireType:
    return WireType(symbol(annotation), None if to_live is None else symbol(to_live))


# Live annotations and their wire representations. A whole annotation is looked
# up before it is decomposed, so a union whose members convert differently can
# be given one explicit conversion.
WIRE_TYPES: Mapping[object, WireType] = {
    ColumnMask: __wire(Mask, live.mask_to_live),
    CompoundColumnMask: __wire(Mask, live.mask_to_live),
    ConstructedColumn: __wire(Expression, live.to_column),
    ConstructedColumn | DerivedScalarValue: __wire(
        Expression, live.to_column_or_scalar
    ),
    Region: __wire(RegionMessage, live.region_to_live),
    u.UnitBase: __wire(UnitString, live.to_unit),
    np.ndarray | DataIndex: __wire(RowIndices, live.to_rows),
    npt.NDArray[np.integer]: __wire(RowIndices, live.to_rows),
    sct.ColumnNames: __wire(scwire.ColumnNames),
    sct.StructureSelection: __wire(scwire.StructureSelection, live.selection_to_live),
    sct.StructureDrop: __wire(scwire.StructureDrop, live.drop_to_live),
    sct.UnitConversions: __wire(scwire.UnitConversions, live.unit_conversions_to_live),
    sct.StructureUnitSettings: __wire(
        scwire.StructureUnitSettings, live.unit_settings_to_live
    ),
    int: WireType(Symbol("pydantic", "StrictInt")),
    float: __wire(FiniteNumber),
    bool: WireType(Symbol("pydantic", "StrictBool")),
    str: __wire(str),
    NoneType: WireType(Symbol("builtins", "None")),
}

# Live values the API accepts that cannot travel over the wire. They are dropped
# from unions; an annotation made only of them cannot be generated.
LIVE_ONLY: frozenset[object] = frozenset({np.ndarray, u.Quantity})

# Annotated metadata that is carried into generated fields unchanged.
METADATA: frozenset[type] = frozenset({Ge})

__PACKAGE = Path(__file__).resolve().parents[1]

SPECS: tuple[MessageSpec, ...] = (
    MessageSpec(
        target=Dataset,
        public_name="opencosmo.Dataset",
        methods=(
            "filter",
            "select",
            "drop",
            "sort_by",
            "take",
            "take_range",
            "take_rows",
            "bound",
            "with_new_columns",
            "with_units",
        ),
        output=__PACKAGE / "messages" / "dataset.py",
        namespace={
            "ColumnMask": ColumnMask,
            "CompoundColumnMask": CompoundColumnMask,
            "ConstructedColumn": ConstructedColumn,
            "DataIndex": DataIndex,
            "DerivedScalarValue": DerivedScalarValue,
            "Ge": Ge,
            "Iterable": Iterable,
            "Region": Region,
        },
        checks={
            "take_range": (Check(symbol(checks.ordered_range), ("start", "end")),),
            "with_new_columns": (
                Check(
                    symbol(checks.described_columns), ("descriptions", "new_columns")
                ),
            ),
        },
    ),
    MessageSpec(
        target=Lightcone,
        public_name="opencosmo.Lightcone",
        methods=(
            "filter",
            "select",
            "drop",
            "sort_by",
            "take",
            "take_range",
            "take_rows",
            "bound",
            "with_new_columns",
            "with_units",
            "with_redshift_range",
            "pixel_search",
        ),
        output=__PACKAGE / "messages" / "lightcone.py",
        namespace={
            "ColumnMask": ColumnMask,
            "CompoundColumnMask": CompoundColumnMask,
            "ConstructedColumn": ConstructedColumn,
            "DataIndex": DataIndex,
            "DerivedScalarValue": DerivedScalarValue,
            "Ge": Ge,
            "Iterable": Iterable,
            "Region": Region,
            "np": np,
            "npt": npt,
            "u": u,
        },
        checks={
            "take_range": (Check(symbol(checks.ordered_range), ("start", "end")),),
            "with_new_columns": (
                Check(symbol(checks.described_columns), ("descriptions", "columns")),
            ),
            "pixel_search": (Check(symbol(checks.pixel_search), ("pixels", "nside")),),
        },
    ),
    MessageSpec(
        target=StructureCollection,
        public_name="opencosmo.StructureCollection",
        methods=(
            "filter",
            "select",
            "drop",
            "sort_by",
            "take",
            "take_range",
            "take_rows",
            "bound",
            "with_new_columns",
            "with_units",
            "with_datasets",
            "with_redshift_range",
            "pixel_search",
        ),
        output=__PACKAGE / "messages" / "structure_collection.py",
        namespace={
            "ColumnNames": sct.ColumnNames,
            "ColumnMask": ColumnMask,
            "CompoundColumnMask": CompoundColumnMask,
            "ConstructedColumn": ConstructedColumn,
            "DataIndex": DataIndex,
            "Ge": Ge,
            "Region": Region,
            "StructureSelection": sct.StructureSelection,
            "StructureDrop": sct.StructureDrop,
            "StructureUnitSettings": sct.StructureUnitSettings,
            "UnitConversions": sct.UnitConversions,
            "np": np,
            "npt": npt,
            "u": u,
        },
        checks={
            "take_range": (Check(symbol(checks.ordered_range), ("start", "end")),),
            "with_new_columns": (
                Check(
                    symbol(checks.described_columns), ("descriptions", "new_columns")
                ),
            ),
            "pixel_search": (Check(symbol(checks.pixel_search), ("pixels", "nside")),),
        },
    ),
)


__all__ = [
    "LIVE_ONLY",
    "METADATA",
    "SPECS",
    "WIRE_TYPES",
    "Check",
    "MessageSpec",
    "Symbol",
    "WireType",
    "symbol",
]
