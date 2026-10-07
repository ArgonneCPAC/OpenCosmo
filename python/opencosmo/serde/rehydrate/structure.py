"""Apply serialized transformation messages to structure collections."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import astropy.units as u
import numpy as np

from opencosmo.column.column import Column, DerivedScalarValue

from ..messages import (
    BoundMessage,
    PixelSearchMessage,
    SortByMessage,
    StructureDropMessage,
    StructureFilterMessage,
    StructureSelectMessage,
    StructureWithNewColumnsMessage,
    StructureWithUnitsMessage,
    TakeMessage,
    TakeRangeMessage,
    TakeRowsMessage,
    WithDatasetsMessage,
    WithRedshiftRangeMessage,
)
from .expression import expression_to_live, mask_to_live
from .region import region_to_live

if TYPE_CHECKING:
    from opencosmo.collection.structure.structure import StructureCollection

    from ..expression import Expression
    from ..messages import (
        StructureMessage,
        StructureSelectionTarget,
        StructureUnitTarget,
    )

type LiveDerivedColumns = dict[str, Column | DerivedScalarValue]
type SelectionLeaf = dict[str, list[str] | LiveDerivedColumns]
type SelectionTree = dict[str, SelectionLeaf | "SelectionTree"]
type DropTree = dict[str, list[str] | "DropTree"]


def apply_structure_message(
    collection: StructureCollection,
    message: StructureMessage,
) -> StructureCollection:
    """Apply a validated transformation message to a structure collection."""
    match message:
        case StructureFilterMessage(masks=masks, on_galaxies=on_galaxies, mode=mode):
            return collection.filter(
                *(mask_to_live(mask) for mask in masks),
                on_galaxies=on_galaxies,
                mode=mode.value,
            )
        case StructureSelectMessage(
            columns=columns,
            derived_columns=derived,
            targets=targets,
            mode=mode,
        ):
            live_derived = _live_derived_columns(derived, allow_scalar=False)
            if not targets:
                return collection.select(
                    *columns,
                    mode=mode.value,
                    **live_derived,  # type: ignore[arg-type]
                )
            selection: SelectionTree = {}
            for path, selection_target in targets.items():
                _merge_tree(selection, _selection_tree(path, selection_target))
            return collection.select(mode=mode.value, **selection)  # type: ignore[arg-type]
        case StructureDropMessage(columns=columns, targets=targets):
            if not targets:
                return collection.drop(*columns)
            dropped: DropTree = {}
            for path, drop_target in targets.items():
                _merge_tree(dropped, _drop_tree(path, list(drop_target.columns)))
            return collection.drop(**dropped)  # type: ignore[arg-type]
        case SortByMessage(column=column, invert=invert):
            if column is None:
                raise ValueError("StructureCollection sorting requires a column")
            return collection.sort_by(column, invert=invert)
        case TakeMessage(n=n, at=at, mode=mode):
            return collection.take(n, at=at.value, mode=mode.value)
        case TakeRangeMessage(start=start, end=end, mode=mode):
            return collection.take_range(start, end, mode=mode.value)
        case TakeRowsMessage(rows=rows):
            return collection.take_rows(np.asarray(rows, dtype=np.int64))
        case BoundMessage(region=region, select_by=select_by):
            return collection.bound(region_to_live(region), select_by=select_by)
        case StructureWithNewColumnsMessage(
            dataset=dataset,
            columns=columns,
            descriptions=descriptions,
            allow_overwrite=allow_overwrite,
            mode=mode,
        ):
            live_columns = _live_derived_columns(columns, allow_scalar=False)
            return collection.with_new_columns(
                dataset,
                descriptions=descriptions,
                allow_overwrite=allow_overwrite,
                mode=mode.value,
                **live_columns,
            )
        case StructureWithUnitsMessage(
            convention=convention,
            conversions=conversions,
            datasets=datasets,
        ):
            return collection.with_units(
                None if convention is None else convention.value,
                {
                    u.Unit(source): u.Unit(target)
                    for source, target in conversions.items()
                },
                **{
                    name: _unit_target_to_live(target)
                    for name, target in datasets.items()
                },
            )
        case WithDatasetsMessage(datasets=datasets):
            return collection.with_datasets(datasets)
        case WithRedshiftRangeMessage(z_low=z_low, z_high=z_high):
            return collection.with_redshift_range(z_low, z_high)
        case PixelSearchMessage(pixels=pixels, nside=nside):
            return collection.pixel_search(
                np.asarray(sorted(pixels), dtype=np.int64), nside=nside
            )
    raise TypeError(f"Unsupported structure message type: {type(message).__name__}")


def _live_derived_columns(
    expressions: dict[str, Expression], *, allow_scalar: bool
) -> LiveDerivedColumns:
    output: LiveDerivedColumns = {}
    for name, expression in expressions.items():
        live = expression_to_live(expression)
        if isinstance(live, Column):
            output[name] = live
        elif allow_scalar and isinstance(live, DerivedScalarValue):
            output[name] = live
        else:
            raise ValueError(f"Expression {name!r} must produce a column")
    return output


def _selection_tree(path: str, target: StructureSelectionTarget) -> SelectionTree:
    leaf: SelectionLeaf = {
        "columns": list(target.columns),
        "derived_columns": _live_derived_columns(
            target.derived_columns, allow_scalar=False
        ),
    }
    parts = path.split(".")
    tree: SelectionTree = {parts[-1]: leaf}
    for part in reversed(parts[:-1]):
        tree = {part: tree}
    return tree


def _drop_tree(path: str, columns: list[str]) -> DropTree:
    parts = path.split(".")
    tree: DropTree = {parts[-1]: columns}
    for part in reversed(parts[:-1]):
        tree = {part: tree}
    return tree


def _merge_tree(
    destination: SelectionTree | DropTree,
    source: SelectionTree | DropTree,
) -> None:
    for key, value in source.items():
        if key not in destination:
            destination[key] = value  # type: ignore[assignment]
            continue
        existing = destination[key]
        if not isinstance(existing, dict) or not isinstance(value, dict):
            raise ValueError(f"Conflicting targets for dataset path {key!r}")
        _merge_tree(
            cast("SelectionTree | DropTree", existing),
            cast("SelectionTree | DropTree", value),
        )


def _unit_target_to_live(
    target: StructureUnitTarget,
) -> dict[str, u.Unit | dict[u.Unit, u.Unit]]:
    values: dict[str, u.Unit | dict[u.Unit, u.Unit]] = {
        name: u.Unit(unit) for name, unit in target.columns.items()
    }
    if target.conversions:
        values["conversions"] = {
            u.Unit(source): u.Unit(destination)
            for source, destination in target.conversions.items()
        }
    return values


__all__ = ["apply_structure_message"]
