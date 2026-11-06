"""Convert validated wire values into live OpenCosmo objects."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, cast

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
from .region import (
    BoxRegionMessage,
    ConeRegionMessage,
    HealpixRegionMessage,
    SkyboxRegionMessage,
)
from .structure import DerivedSelection

if TYPE_CHECKING:
    from collections.abc import Callable

    from opencosmo.collection.structure import types as sct
    from opencosmo.spatial.protocols import Region

    from . import structure as scwire
    from .expression import Expression, Mask
    from .region import RegionMessage

type LiveScalar = int | float | u.Quantity
type LiveExpression = Column | DerivedScalarValue | LiveScalar
type LiveMask = ColumnMask | CompoundColumnMask


def expression_to_live(expression: Expression) -> LiveExpression:
    """Build a lazy OpenCosmo expression from a serialized expression."""
    match expression:
        case ColumnReference(name=name):
            return col(name)
        case NumberLiteral(value=value):
            return value
        case QuantityLiteral(value=value, unit=unit):
            return u.Quantity(value, u.Unit(unit))
        case ArithmeticExpression(operator=operation, left=left, right=right):
            return __arithmetic(
                operation,
                expression_to_live(left),
                expression_to_live(right),
            )
        case UnaryExpression(operator=operation, operand=operand):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError(f"{operation.value} requires a vector expression")
            return __unary(operation, live_operand)
        case Arctan2Expression(left=left, right=right):
            live_left = expression_to_live(left)
            if not isinstance(live_left, Column):
                raise TypeError("arctan2 requires a vector expression on the left")
            return live_left.arctan2(expression_to_live(right))
        case ReductionExpression(operator=operation, operand=operand):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError(f"{operation.value} requires a vector expression")
            return __reduce(operation, live_operand)
        case QuantileExpression(operand=operand, quantile=quantile):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError("quantile requires a vector expression")
            return live_operand.quantile(quantile)
    raise TypeError(f"Unsupported expression type: {type(expression).__name__}")


def mask_to_live(mask: Mask) -> LiveMask:
    """Build a lazy OpenCosmo mask from a serialized mask."""
    match mask:
        case ComparisonMask(operator=operation, left=left, right=right):
            live_left = expression_to_live(left)
            live_right = expression_to_live(right)
            if not isinstance(live_left, Column) and not isinstance(live_right, Column):
                raise TypeError("A comparison mask requires a vector expression")
            return ColumnMask(
                live_left,  # type: ignore[arg-type]
                live_right,  # type: ignore[arg-type]
                __comparison_operator(operation),  # type: ignore[arg-type]
            )
        case MembershipMask(operand=operand, values=values):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError("A membership mask requires a vector expression")
            live_values = tuple(expression_to_live(value) for value in values)
            return live_operand.isin(live_values)
        case CompoundMask(operator=operation, left=left, right=right):
            live_left = mask_to_live(left)
            live_right = mask_to_live(right)
            if operation is BooleanOperator.AND:
                return live_left & live_right
            return live_left | live_right
    raise TypeError(f"Unsupported mask type: {type(mask).__name__}")


def __arithmetic(
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


def __unary(operation: UnaryOperator, operand: Column) -> Column:
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


def __reduce(operation: ReductionOperator, operand: Column) -> DerivedScalarValue:
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


def __comparison_operator(
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


def region_to_live(region: RegionMessage) -> Region:
    """Build an OpenCosmo region from a serialized region message."""
    match region:
        case BoxRegionMessage(p1=p1, p2=p2):
            return make_box(p1, p2)
        case ConeRegionMessage(center=center, radius=radius):
            return make_cone(center, radius)
        case SkyboxRegionMessage(p1=p1, p2=p2):
            return make_skybox(p1, p2)
        case HealpixRegionMessage(pixels=pixels, nside=nside):
            return HealpixRegion(np.asarray(sorted(pixels), dtype=np.int64), nside)
    raise TypeError(f"Unsupported region type: {type(region).__name__}")


def to_column(expression: Expression) -> Column:
    """Build a lazy column, rejecting expressions that do not produce one."""
    live = expression_to_live(expression)
    if not isinstance(live, Column):
        raise TypeError("Expression must produce a column")
    return live


def to_column_or_scalar(expression: Expression) -> Column | DerivedScalarValue:
    """Build a lazy column or scalar reduction, rejecting plain literals."""
    live = expression_to_live(expression)
    if not isinstance(live, Column | DerivedScalarValue):
        raise TypeError("Expression must produce a column or a scalar reduction")
    return live


def to_unit(unit: str) -> u.UnitBase:
    """Build an Astropy unit from a normalized unit string."""
    return u.Unit(unit)


def to_rows(rows: tuple[int, ...]) -> np.ndarray:
    """Build a signed 64-bit row index from row positions."""
    return np.asarray(rows, dtype=np.int64)


def selection_to_live(selection: scwire.StructureSelection) -> sct.StructureSelection:
    """Convert explicit, automatic, and nested column selections recursively."""
    if isinstance(selection, str | tuple):
        return selection
    if isinstance(selection, DerivedSelection):
        return {
            "columns": selection.columns,
            "derived_columns": {
                name: to_column(expression)
                for name, expression in selection.derived_columns.items()
            },
        }
    if isinstance(selection, dict):
        return {name: selection_to_live(value) for name, value in selection.items()}
    return to_column(selection)


def drop_to_live(columns: scwire.StructureDrop) -> sct.StructureDrop:
    """Convert nested drop specifications into live dictionaries."""
    if isinstance(columns, str | tuple):
        return columns
    return {name: drop_to_live(value) for name, value in columns.items()}


def unit_conversions_to_live(
    conversions: scwire.UnitConversions,
) -> sct.UnitConversions:
    """Convert both sides of a blanket unit conversion."""
    return {to_unit(source): to_unit(target) for source, target in conversions.items()}


def unit_settings_to_live(
    settings: scwire.StructureUnitSettings,
) -> sct.StructureUnitSettings:
    """Convert column units and blanket conversions at every collection depth."""
    result: sct.StructureUnitSettings = {}
    for name, value in settings.items():
        if name == "conversions":
            result[name] = unit_conversions_to_live(
                cast("scwire.UnitConversions", value)
            )
        elif isinstance(value, str):
            result[name] = to_unit(value)
        else:
            result[name] = unit_settings_to_live(value)
    return result


__all__ = [
    "LiveExpression",
    "LiveMask",
    "LiveScalar",
    "drop_to_live",
    "expression_to_live",
    "mask_to_live",
    "region_to_live",
    "selection_to_live",
    "to_column",
    "to_column_or_scalar",
    "to_rows",
    "to_unit",
    "unit_conversions_to_live",
    "unit_settings_to_live",
]
