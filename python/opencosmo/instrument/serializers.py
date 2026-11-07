from __future__ import annotations

import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opencosmo.column.column import EvaluatedColumn
    from opencosmo.instrument.messages import (
        DatasetEvaluateMessage,
        DatasetWithDataColumnsMessage,
    )
    from opencosmo.serde import (
        DatasetBoundMessage,
        DatasetDropMessage,
        DatasetFilterMessage,
        DatasetSelectMessage,
        DatasetTakeMessage,
        DatasetTakeRangeMessage,
        DatasetTakeRowsMessage,
        DatasetWithNewColumnsMessage,
    )


# serde.dispatch imports Dataset, and dataset.operations imports this module,
# so serde must be imported at call time.
def serialize_filter(masks, mode) -> DatasetFilterMessage:
    from opencosmo import serde
    from opencosmo.serde.live import mask_from_live

    masks = tuple(mask_from_live(mask) for mask in masks)
    return serde.DatasetFilterMessage(masks=masks, mode=mode)


def serialize_take_range(start, end, mode) -> DatasetTakeRangeMessage:
    from opencosmo.serde import DatasetTakeRangeMessage

    # internal callers pass NumPy integers; index() accepts those but not floats
    return DatasetTakeRangeMessage(
        start=operator.index(start), end=operator.index(end), mode=mode
    )


def serialize_take(n, at, mode) -> DatasetTakeMessage:
    from opencosmo.serde import DatasetTakeMessage

    return DatasetTakeMessage(n=operator.index(n), at=at, mode=mode)


def __columns_to_wire(
    columns: tuple[str | Iterable[str], ...],
) -> tuple[str | tuple[str, ...], ...]:
    # select/drop treat each group as a set, so sort for a deterministic record
    return tuple(
        group if isinstance(group, str) else tuple(sorted(group)) for group in columns
    )


def serialize_select(columns, mode, derived_columns) -> DatasetSelectMessage:
    from opencosmo import serde
    from opencosmo.serde.live import from_column_or_scalar

    return serde.DatasetSelectMessage(
        columns=__columns_to_wire(columns),
        mode=mode,
        derived_columns={
            name: from_column_or_scalar(value)
            for name, value in derived_columns.items()
        },
    )


def serialize_drop(columns) -> DatasetDropMessage:
    from opencosmo import serde

    return serde.DatasetDropMessage(columns=__columns_to_wire(columns))


def serialize_with_new_columns(
    descriptions, new_columns, **kwargs
) -> DatasetWithNewColumnsMessage | DatasetWithDataColumnsMessage:
    import numpy as np

    from opencosmo import serde
    from opencosmo.serde.live import from_column

    data_columns = tuple(
        sorted(
            name
            for name, value in new_columns.items()
            if isinstance(value, np.ndarray)  # includes astropy Quantity
        )
    )
    if isinstance(descriptions, dict):
        descriptions = {
            name: text
            for name, text in descriptions.items()
            if name not in data_columns
        }
    call = serde.DatasetWithNewColumnsMessage(
        descriptions=descriptions,
        new_columns={
            name: from_column(value)
            for name, value in new_columns.items()
            if name not in data_columns
        },
        **kwargs,
    )
    if not data_columns:
        return call

    from opencosmo.instrument.messages import DatasetWithDataColumnsMessage

    return DatasetWithDataColumnsMessage(call=call, data_columns=data_columns)


def serialize_take_rows(rows) -> DatasetTakeRowsMessage:
    from opencosmo import serde
    from opencosmo.serde.live import from_rows

    return serde.DatasetTakeRowsMessage(rows=from_rows(rows))


def serialize_bound(region, select_by) -> DatasetBoundMessage:
    from opencosmo import serde
    from opencosmo.serde.live import region_from_live

    return serde.DatasetBoundMessage(
        region=region_from_live(region), select_by=select_by
    )


def serialize_evaluate(column: EvaluatedColumn) -> DatasetEvaluateMessage:
    from opencosmo.instrument.messages import DatasetEvaluateMessage

    return DatasetEvaluateMessage(
        requires=tuple(sorted(column.requires_names)),
        produces=tuple(sorted(column.produces)),
    )
