"""Apply serialized transformation messages to HEALPix maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo.column.column import Column, DerivedScalarValue

from ..messages import (
    DropMessage,
    FilterMessage,
    HealpixBoundMessage,
    SelectMessage,
    SortByMessage,
    TakeMessage,
    TakeRangeMessage,
    TakeRowsMessage,
    WithNewColumnsMessage,
)
from .expression import expression_to_live, mask_to_live
from .region import region_to_live

if TYPE_CHECKING:
    from opencosmo.collection.lightcone.healpix_map import HealpixMap

    from ..messages import HealpixMapMessage


def apply_healpix_map_message(
    healpix_map: HealpixMap,
    message: HealpixMapMessage,
) -> HealpixMap:
    """Apply a validated transformation message to a HEALPix map."""
    match message:
        case FilterMessage(masks=masks, mode=mode):
            return healpix_map.filter(  # type: ignore[arg-type]
                *(mask_to_live(mask) for mask in masks),  # type: ignore[arg-type]
                mode=mode.value,
            )
        case SelectMessage(columns=columns, derived_columns=derived, mode=mode):
            if mode.value != "global":
                raise ValueError("HealpixMap select does not support local mode")
            live_derived: dict[str, Column | DerivedScalarValue] = {}
            for name, expression in derived.items():
                live = expression_to_live(expression)
                if not isinstance(live, (Column, DerivedScalarValue)):
                    raise ValueError(
                        f"Selected expression {name!r} must depend on a column"
                    )
                live_derived[name] = live
            return healpix_map.select(*columns, **live_derived)
        case DropMessage(columns=columns):
            return healpix_map.drop(columns)
        case SortByMessage(column=column, invert=invert):
            if column is None:
                raise ValueError("HealpixMap sorting requires a column")
            return healpix_map.sort_by(column, invert=invert)
        case TakeMessage(n=n, at=at, mode=mode):
            if mode.value != "local":
                raise ValueError("HealpixMap take does not support global mode")
            return healpix_map.take(n, at=at.value)
        case TakeRangeMessage(start=start, end=end, mode=mode):
            if mode.value != "local":
                raise ValueError("HealpixMap take_range does not support global mode")
            return healpix_map.take_range(start, end)
        case TakeRowsMessage(rows=rows):
            if not rows:
                raise ValueError("HealpixMap take_rows requires at least one row")
            return healpix_map.take_rows(np.asarray(rows, dtype=np.int64))
        case HealpixBoundMessage(region=region, inclusive=inclusive):
            return healpix_map.bound(
                region_to_live(region),
                inclusive=inclusive,
            )
        case WithNewColumnsMessage(
            columns=columns,
            descriptions=descriptions,
            allow_overwrite=allow_overwrite,
            mode=mode,
        ):
            if allow_overwrite:
                raise ValueError(
                    "HealpixMap with_new_columns does not support overwrite"
                )
            if mode.value != "global":
                raise ValueError(
                    "HealpixMap with_new_columns does not support local mode"
                )
            live_columns: dict[str, Column] = {}
            for name, expression in columns.items():
                live = expression_to_live(expression)
                if not isinstance(live, Column):
                    raise ValueError(
                        f"New column expression {name!r} must produce a column"
                    )
                live_columns[name] = live
            return healpix_map.with_new_columns(
                descriptions=descriptions,
                **live_columns,
            )
    raise TypeError(f"Unsupported HEALPix map message type: {type(message).__name__}")


__all__ = ["apply_healpix_map_message"]
