"""Translation and application of validated serialization messages."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from opencosmo.column.column import (
    Column,
    ColumnMask,
    CompoundColumnMask,
    DerivedScalarValue,
    col,
)
from opencosmo.spatial.builders import make_box, make_cone, make_skybox
from opencosmo.spatial.region import HealpixRegion

from .expression import (
    Arctan2Expression,
    ArithmeticExpression,
    ArithmeticOperator,
    BooleanOperator,
    ColumnReference,
    ComparisonMask,
    ComparisonOperator,
    CompoundMask,
    MembershipMask,
    NumberLiteral,
    QuantileExpression,
    QuantityLiteral,
    ReductionExpression,
    ReductionOperator,
    UnaryExpression,
    UnaryOperator,
)
from .message import (
    BoundMessage,
    BoxRegionMessage,
    ConeRegionMessage,
    DropMessage,
    FilterMessage,
    HealpixRegionMessage,
    SelectMessage,
    SkyboxRegionMessage,
    SortByMessage,
    TakeMessage,
    TakeRangeMessage,
    TakeRowsMessage,
    WithNewColumnsMessage,
    WithUnitsMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opencosmo.dataset import Dataset
    from opencosmo.spatial.protocols import Region

    from .expression import Expression, Mask
    from .message import DatasetMessage, RegionMessage

type LiveScalar = int | float | u.Quantity
type LiveExpression = Column | DerivedScalarValue | LiveScalar
type LiveMask = ColumnMask | CompoundColumnMask


def expression_to_live(expression: Expression) -> LiveExpression:
    """Build a lazy OpenCosmo expression from a serialized expression.

    Parameters
    ----------
    expression
        Validated expression to translate.

    Returns
    -------
    Column, DerivedScalarValue, int, float, or astropy.units.Quantity
        The corresponding lazy expression or scalar literal.

    Raises
    ------
    ValueError
        If an operation requires a vector expression but receives a scalar.
    TypeError
        If an unsupported expression model is provided.
    """
    match expression:
        case ColumnReference(name=name):
            return col(name)
        case NumberLiteral(value=value):
            return value
        case QuantityLiteral(value=value, unit=unit):
            return u.Quantity(value, u.Unit(unit))
        case ArithmeticExpression(operator=operation, left=left, right=right):
            return _arithmetic(
                operation,
                expression_to_live(left),
                expression_to_live(right),
            )
        case UnaryExpression(operator=operation, operand=operand):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise ValueError(f"{operation.value} requires a vector expression")
            return _unary(operation, live_operand)
        case Arctan2Expression(left=left, right=right):
            live_left = expression_to_live(left)
            if not isinstance(live_left, Column):
                raise ValueError("arctan2 requires a vector expression on the left")
            return live_left.arctan2(expression_to_live(right))
        case ReductionExpression(operator=operation, operand=operand):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise ValueError(f"{operation.value} requires a vector expression")
            return _reduce(operation, live_operand)
        case QuantileExpression(operand=operand, quantile=quantile):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise ValueError("quantile requires a vector expression")
            return live_operand.quantile(quantile)
    raise TypeError(f"Unsupported expression type: {type(expression).__name__}")


def mask_to_live(mask: Mask) -> LiveMask:
    """Build a lazy OpenCosmo mask from a serialized mask.

    Parameters
    ----------
    mask
        Validated mask to translate.

    Returns
    -------
    ColumnMask or CompoundColumnMask
        The corresponding lazy mask.

    Raises
    ------
    ValueError
        If a mask cannot produce one value per dataset row.
    TypeError
        If an unsupported mask model is provided.
    """
    match mask:
        case ComparisonMask(operator=operation, left=left, right=right):
            live_left = expression_to_live(left)
            live_right = expression_to_live(right)
            if not isinstance(live_left, Column) and not isinstance(live_right, Column):
                raise ValueError("A comparison mask requires a vector expression")
            return ColumnMask(
                live_left,  # type: ignore[arg-type]
                live_right,  # type: ignore[arg-type]
                _comparison_operator(operation),  # type: ignore[arg-type]
            )
        case MembershipMask(operand=operand, values=values):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise ValueError("A membership mask requires a vector expression")
            live_values = tuple(expression_to_live(value) for value in values)
            return live_operand.isin(live_values)
        case CompoundMask(operator=operation, left=left, right=right):
            live_left = mask_to_live(left)
            live_right = mask_to_live(right)
            if operation is BooleanOperator.AND:
                return live_left & live_right
            return live_left | live_right
    raise TypeError(f"Unsupported mask type: {type(mask).__name__}")


def apply_message(dataset: Dataset, message: DatasetMessage) -> Dataset:
    """Apply a validated operation message to a dataset.

    Parameters
    ----------
    dataset
        Dataset on which to perform the operation.
    message
        Validated filter or select message.

    Returns
    -------
    Dataset
        The transformed dataset.

    Raises
    ------
    TypeError
        If ``message`` is not a supported dataset message.
    ValueError
        If a selected expression is only a scalar literal.
    """
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
            return dataset.bound(_region_to_live(region), select_by=select_by)
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
    raise TypeError(f"Unsupported dataset message type: {type(message).__name__}")


def _region_to_live(region: RegionMessage) -> Region:
    match region:
        case BoxRegionMessage(p1=p1, p2=p2):
            return make_box(p1, p2)
        case ConeRegionMessage(center=center, radius=radius):
            return make_cone(center, radius)
        case SkyboxRegionMessage(p1=p1, p2=p2):
            return make_skybox(p1, p2)
        case HealpixRegionMessage(pixels=pixels, nside=nside):
            return HealpixRegion(np.asarray(sorted(pixels), dtype=np.int64), nside)


def _arithmetic(
    operation: ArithmeticOperator,
    left: LiveExpression,
    right: LiveExpression,
) -> LiveExpression:
    operations = {
        ArithmeticOperator.ADD: operator.add,
        ArithmeticOperator.SUBTRACT: operator.sub,
        ArithmeticOperator.MULTIPLY: operator.mul,
        ArithmeticOperator.DIVIDE: operator.truediv,
        ArithmeticOperator.POWER: operator.pow,
    }
    function = operations[operation]
    if isinstance(left, Column) or isinstance(right, Column):
        return Column(left, right, function)
    if isinstance(left, DerivedScalarValue) or isinstance(right, DerivedScalarValue):
        return DerivedScalarValue(left, right, function)
    return function(left, right)


def _unary(operation: UnaryOperator, operand: Column) -> Column:
    match operation:
        case UnaryOperator.LOG10:
            return operand.log10()
        case UnaryOperator.EXP10:
            return operand.exp10()
        case UnaryOperator.SQRT:
            return operand.sqrt()
        case UnaryOperator.ARCSIN:
            return operand.arcsin()
        case UnaryOperator.ARCCOS:
            return operand.arccos()


def _reduce(operation: ReductionOperator, operand: Column) -> DerivedScalarValue:
    match operation:
        case ReductionOperator.MEAN:
            return operand.mean()
        case ReductionOperator.STANDARD_DEVIATION:
            return operand.std()
        case ReductionOperator.VARIANCE:
            return operand.var()
        case ReductionOperator.MINIMUM:
            return operand.min()
        case ReductionOperator.MAXIMUM:
            return operand.max()
        case ReductionOperator.MEDIAN:
            return operand.median()
        case ReductionOperator.SUM:
            return operand.sum()


def _comparison_operator(
    operation: ComparisonOperator,
) -> Callable[[LiveExpression, LiveExpression], bool]:
    operations: dict[
        ComparisonOperator, Callable[[LiveExpression, LiveExpression], bool]
    ] = {
        ComparisonOperator.EQUAL: operator.eq,
        ComparisonOperator.NOT_EQUAL: operator.ne,
        ComparisonOperator.GREATER_THAN: operator.gt,
        ComparisonOperator.GREATER_THAN_OR_EQUAL: operator.ge,
        ComparisonOperator.LESS_THAN: operator.lt,
        ComparisonOperator.LESS_THAN_OR_EQUAL: operator.le,
    }
    return operations[operation]


__all__ = [
    "LiveExpression",
    "LiveMask",
    "LiveScalar",
    "apply_message",
    "expression_to_live",
    "mask_to_live",
]
