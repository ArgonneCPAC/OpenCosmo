"""Apply serialized transformation messages to simulation collections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u

from opencosmo.column.column import Column

from ..messages import (
    BoundMessage,
    ClearMatchMessage,
    DropMessage,
    FilterMessage,
    MatchMessage,
    SelectMessage,
    SimulationWithNewColumnsMessage,
    SortByMessage,
    StructureDropMessage,
    StructureFilterMessage,
    StructureSelectMessage,
    StructureWithNewColumnsMessage,
    StructureWithUnitsMessage,
    TakeMessage,
    TakeRangeMessage,
    WithNewColumnsMessage,
    WithUnitsMessage,
)
from .expression import expression_to_live, mask_to_live
from .region import region_to_live

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opencosmo.collection.simulation.simulation import SimulationCollection

    from ..expression import Expression
    from ..messages import SimulationMessage


def apply_simulation_message(
    collection: SimulationCollection,
    message: SimulationMessage,
) -> SimulationCollection:
    """Apply a validated transformation message to a simulation collection."""
    match message:
        case MatchMessage(dataset=dataset):
            return collection.match(dataset)
        case ClearMatchMessage():
            return collection.clear_match()
        case FilterMessage(masks=masks, mode=mode):
            return collection.filter(
                *(mask_to_live(mask) for mask in masks),  # type: ignore[arg-type]
                mode=mode.value,
            )
        case StructureFilterMessage(masks=masks, on_galaxies=on_galaxies, mode=mode):
            return collection.filter(
                *(mask_to_live(mask) for mask in masks),  # type: ignore[arg-type]
                on_galaxies=on_galaxies,
                mode=mode.value,
            )
        case SelectMessage(columns=columns, derived_columns=derived, mode=mode):
            live = _live_columns(derived)
            return collection.select(*columns, mode=mode.value, **live)
        case StructureSelectMessage(
            columns=columns,
            derived_columns=derived,
            targets=targets,
            mode=mode,
        ):
            live = _live_columns(derived)
            select_kwargs: dict[str, str | list[str] | dict] = dict(live)  # type: ignore[arg-type]
            for path, selection_target in targets.items():
                if "." in path:
                    raise ValueError(
                        "SimulationCollection structure selection targets must be direct dataset names"
                    )
                select_kwargs[path] = {
                    "columns": list(selection_target.columns),
                    "derived_columns": _live_columns(selection_target.derived_columns),
                }
            return collection.select(*columns, mode=mode.value, **select_kwargs)
        case DropMessage(columns=columns):
            return collection.drop(*columns)
        case StructureDropMessage(columns=columns, targets=targets):
            drop_kwargs: dict[str, list[str]] = {}
            for path, drop_target in targets.items():
                if "." in path:
                    raise ValueError(
                        "SimulationCollection structure drop targets must be direct dataset names"
                    )
                drop_kwargs[path] = list(drop_target.columns)
            return collection.drop(*columns, **drop_kwargs)
        case SortByMessage(column=column, invert=invert):
            if column is None:
                raise ValueError("SimulationCollection sorting requires a column")
            return collection.sort_by(column, invert=invert)
        case TakeMessage(n=n, at=at, mode=mode):
            return collection.take(n, at=at.value, mode=mode.value)
        case TakeRangeMessage(start=start, end=end, mode=mode):
            return collection.take_range(start, end, mode=mode.value)
        case BoundMessage(region=region, select_by=select_by):
            return collection.bound(region_to_live(region), select_by=select_by)
        case WithUnitsMessage(
            convention=convention,
            conversions=conversions,
            columns=columns,
        ):
            return collection.with_units(
                None if convention is None else convention.value,
                conversions={
                    u.Unit(source): u.Unit(target)
                    for source, target in conversions.items()
                },
                **{name: u.Unit(unit) for name, unit in columns.items()},
            )
        case StructureWithUnitsMessage():
            raise ValueError(
                "Structure-specific unit targets cannot be delegated uniformly by SimulationCollection"
            )
        case WithNewColumnsMessage(
            columns=columns,
            descriptions=descriptions,
            allow_overwrite=allow_overwrite,
        ):
            return collection.with_new_columns(  # type: ignore[arg-type]
                descriptions=descriptions,
                allow_overwrite=allow_overwrite,
                **_live_columns(columns),  # type: ignore[arg-type]
            )
        case StructureWithNewColumnsMessage(
            dataset=dataset,
            columns=columns,
            descriptions=descriptions,
            allow_overwrite=allow_overwrite,
        ):
            return collection.with_new_columns(  # type: ignore[arg-type]
                dataset,
                descriptions=descriptions,
                allow_overwrite=allow_overwrite,
                **_live_columns(columns),  # type: ignore[arg-type]
            )
        case SimulationWithNewColumnsMessage(
            dataset=dataset,
            datasets=datasets,
            descriptions=descriptions,
            allow_overwrite=allow_overwrite,
            columns=columns,
        ):
            args = () if dataset is None else (dataset,)
            return collection.with_new_columns(
                *args,
                datasets=datasets,
                descriptions=descriptions,
                allow_overwrite=allow_overwrite,
                **_live_columns(columns),
            )
    raise TypeError(f"Unsupported simulation message type: {type(message).__name__}")


def _live_columns(expressions: Mapping[str, Expression]) -> dict[str, Column]:
    output: dict[str, Column] = {}
    for name, expression in expressions.items():
        live = expression_to_live(expression)
        if not isinstance(live, Column):
            raise ValueError(f"Expression {name!r} must produce a column")
        output[name] = live
    return output


__all__ = ["apply_simulation_message"]
