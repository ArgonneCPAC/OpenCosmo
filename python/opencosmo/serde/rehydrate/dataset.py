"""Apply serialized transformation messages to datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from opencosmo.column.column import Column, DerivedScalarValue

from ..messages import (
    BoundMessage,
    DropMessage,
    FilterMessage,
    PixelSearchMessage,
    SelectMessage,
    SortByMessage,
    TakeMessage,
    TakeRangeMessage,
    TakeRowsMessage,
    WithNewColumnsMessage,
    WithRedshiftRangeMessage,
    WithUnitsMessage,
)
from .expression import expression_to_live, mask_to_live
from .region import region_to_live

if TYPE_CHECKING:
    from opencosmo.dataset import Dataset

    from ..messages import DatasetMessage, LightconeMessage


def apply_dataset_message(
    dataset: Dataset,
    message: DatasetMessage | LightconeMessage,
) -> Dataset:
    """Apply a validated transformation message to a dataset."""
    match message:
        case FilterMessage(masks=masks, mode=mode):
            return dataset.filter(
                *(mask_to_live(mask) for mask in masks),  # type: ignore[arg-type]
                mode=mode.value,
            )
        case SelectMessage(columns=columns, derived_columns=derived, mode=mode):
            live_derived: dict[str, Column | DerivedScalarValue] = {}
            for name, expression in derived.items():
                live_expression = expression_to_live(expression)
                if not isinstance(live_expression, (Column, DerivedScalarValue)):
                    raise ValueError(
                        f"Selected expression {name!r} must depend on a column"
                    )
                live_derived[name] = live_expression
            return dataset.select(*columns, mode=mode.value, **live_derived)
        case DropMessage(columns=columns):
            return dataset.drop(*columns)
        case SortByMessage(column=column, invert=invert):
            return dataset.sort_by(column, invert=invert)
        case TakeMessage(n=n, at=at, mode=mode):
            return dataset.take(n, at=at.value, mode=mode.value)
        case TakeRangeMessage(start=start, end=end, mode=mode):
            return dataset.take_range(start, end, mode=mode.value)
        case TakeRowsMessage(rows=rows):
            return dataset.take_rows(np.asarray(rows, dtype=np.int64))
        case BoundMessage(region=region, select_by=select_by):
            return dataset.bound(region_to_live(region), select_by=select_by)
        case WithNewColumnsMessage(
            columns=columns,
            descriptions=descriptions,
            allow_overwrite=allow_overwrite,
            mode=mode,
        ):
            live_columns: dict[str, Column] = {}
            for name, expression in columns.items():
                live_expression = expression_to_live(expression)
                if not isinstance(live_expression, Column):
                    raise ValueError(
                        f"New column expression {name!r} must produce a column"
                    )
                live_columns[name] = live_expression
            return dataset.with_new_columns(
                descriptions=descriptions,
                allow_overwrite=allow_overwrite,
                mode=mode.value,
                **live_columns,
            )
        case WithUnitsMessage(
            convention=convention,
            conversions=conversions,
            columns=columns,
        ):
            return dataset.with_units(
                convention=None if convention is None else convention.value,
                conversions={
                    u.Unit(source): u.Unit(target)
                    for source, target in conversions.items()
                },
                **{name: u.Unit(unit) for name, unit in columns.items()},
            )
        case WithRedshiftRangeMessage() | PixelSearchMessage():
            raise TypeError(
                f"Message type {type(message).__name__} is only supported by Lightcone"
            )
    raise TypeError(f"Unsupported dataset message type: {type(message).__name__}")


__all__ = ["apply_dataset_message"]
