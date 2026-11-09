"""Convert between validated wire values and live OpenCosmo objects."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, cast

import astropy.units as u
import numpy as np

from opencosmo.column.column import (
    Column,
    ColumnMask,
    CompoundColumnMask,
    DerivedScalarValue,
    _arccos,
    _arcsin,
    _arctan2,
    _exp10,
    _log10,
    _max,
    _mean,
    _median,
    _min,
    _quantile,
    _sqrt,
    _std,
    _sum,
    _var,
    col,
    ident,
)
from opencosmo.index import into_array
from opencosmo.spatial.builders import make_box, make_cone, make_skybox
from opencosmo.spatial.region import (
    BoxRegion,
    ConeRegion,
    HealpixRegion,
    SkyboxRegion,
)

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
    from opencosmo.column.column import ExpressionOperand, MaskOperator
    from opencosmo.index import DataIndex
    from opencosmo.spatial.protocols import Region

    from . import structure as scwire
    from .expression import Expression, Mask, ScalarLiteral
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
            return ColumnMask(live_left, live_right, __comparison_operator(operation))
        case MembershipMask(operand=operand, values=values):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError("A membership mask requires a vector expression")
            return live_operand.isin(tuple(__scalar_to_live(value) for value in values))
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


def __scalar_to_live(value: ScalarLiteral) -> LiveScalar:
    match value:
        case NumberLiteral(value=number):
            return number
        case QuantityLiteral(value=number, unit=unit):
            return u.Quantity(number, u.Unit(unit))


def __comparison_operator(operation: ComparisonOperator) -> MaskOperator:
    operations: dict[ComparisonOperator, MaskOperator] = {
        ComparisonOperator.EQUAL: operator.eq,
        ComparisonOperator.NOT_EQUAL: operator.ne,
        ComparisonOperator.GREATER_THAN: operator.gt,
        ComparisonOperator.GREATER_THAN_OR_EQUAL: operator.ge,
        ComparisonOperator.LESS_THAN: operator.lt,
        ComparisonOperator.LESS_THAN_OR_EQUAL: operator.le,
    }
    return operations[operation]


__ARITHMETIC_OPERATORS: dict[Callable[..., object], ArithmeticOperator] = {
    operator.add: ArithmeticOperator.ADD,
    operator.sub: ArithmeticOperator.SUBTRACT,
    operator.mul: ArithmeticOperator.MULTIPLY,
    operator.truediv: ArithmeticOperator.DIVIDE,
    operator.pow: ArithmeticOperator.POWER,
}

__UNARY_OPERATORS: dict[Callable[..., object], UnaryOperator] = {
    _log10: UnaryOperator.LOG10,
    _exp10: UnaryOperator.EXP10,
    _sqrt: UnaryOperator.SQRT,
    _arcsin: UnaryOperator.ARCSIN,
    _arccos: UnaryOperator.ARCCOS,
}

__REDUCTION_OPERATORS: dict[Callable[..., object], ReductionOperator] = {
    _mean: ReductionOperator.MEAN,
    _std: ReductionOperator.STANDARD_DEVIATION,
    _var: ReductionOperator.VARIANCE,
    _min: ReductionOperator.MINIMUM,
    _max: ReductionOperator.MAXIMUM,
    _median: ReductionOperator.MEDIAN,
    _sum: ReductionOperator.SUM,
}

__COMPARISON_OPERATORS: dict[Callable[..., object], ComparisonOperator] = {
    operator.eq: ComparisonOperator.EQUAL,
    operator.ne: ComparisonOperator.NOT_EQUAL,
    operator.gt: ComparisonOperator.GREATER_THAN,
    operator.ge: ComparisonOperator.GREATER_THAN_OR_EQUAL,
    operator.lt: ComparisonOperator.LESS_THAN,
    operator.le: ComparisonOperator.LESS_THAN_OR_EQUAL,
}

__BOOLEAN_OPERATORS: dict[Callable[..., object], BooleanOperator] = {
    operator.and_: BooleanOperator.AND,
    operator.or_: BooleanOperator.OR,
}

# Keyword arguments that the wire grammar implies for parameterized operations.
__DEFAULT_KEYWORDS: dict[Callable[..., object], dict[str, object]] = {
    _log10: {"unit_container": u.DexUnit},
    _exp10: {"expected_unit_container": u.DexUnit},
}


def expression_from_live(expression: LiveExpression | str) -> Expression:
    """Build a serialized expression from a lazy OpenCosmo expression.

    Raises
    ------
    TypeError
        If the expression contains a value or operation with no wire form.
    ValueError
        If the expression contains a non-scalar quantity or an operation
        configured with arguments the wire grammar cannot express.
    """
    match expression:
        case str():
            return ColumnReference(name=expression)
        case Column(lhs=str() as name, rhs=None, operation=operation) if (
            operation is ident
        ):
            return ColumnReference(name=name)
        case Column() | DerivedScalarValue():
            return __node_from_live(expression)
        case u.Quantity() | int() | float() | np.integer() | np.floating():
            return __scalar_from_live(expression)
    raise TypeError(f"Unsupported expression type: {type(expression).__name__}")


def mask_from_live(mask: LiveMask) -> Mask:
    """Build a serialized mask from a lazy OpenCosmo mask.

    Raises
    ------
    TypeError
        If the mask contains a value or operation with no wire form.
    """
    match mask:
        case ColumnMask(left=left, right=right, operator=operation):
            if operation is np.isin:
                if not isinstance(right, Iterable):
                    raise TypeError("A membership mask requires a collection of values")
                return MembershipMask(
                    operand=expression_from_live(left),
                    values=tuple(__scalar_from_live(value) for value in right),
                )
            if isinstance(right, Iterable) and not isinstance(right, u.Quantity):
                raise TypeError("Only membership masks compare against a collection")
            if (comparison := __COMPARISON_OPERATORS.get(operation)) is None:
                raise TypeError(f"Unsupported comparison operation: {operation!r}")
            return ComparisonMask(
                operator=comparison,
                left=expression_from_live(left),
                right=expression_from_live(right),
            )
        case CompoundColumnMask(left=left, right=right, op=operation):
            if (boolean := __BOOLEAN_OPERATORS.get(operation)) is None:
                raise TypeError(f"Unsupported mask combination: {operation!r}")
            return CompoundMask(
                operator=boolean,
                left=mask_from_live(left),
                right=mask_from_live(right),
            )
    raise TypeError(f"Unsupported mask type: {type(mask).__name__}")


def __node_from_live(node: Column | DerivedScalarValue) -> Expression:
    function, keywords = __unwrap(node.operation)
    if (arithmetic := __ARITHMETIC_OPERATORS.get(function)) is not None:
        return ArithmeticExpression(
            operator=arithmetic,
            left=expression_from_live(node.lhs),
            right=expression_from_live(__require_rhs(node)),
        )
    if function is _arctan2:
        return Arctan2Expression(
            left=expression_from_live(node.lhs),
            right=expression_from_live(__require_rhs(node)),
        )
    if function is _quantile:
        if not isinstance(quantile := keywords.get("q"), float | int):
            raise TypeError(f"Unsupported quantile: {quantile!r}")
        return QuantileExpression(
            operand=expression_from_live(node.lhs), quantile=float(quantile)
        )
    if keywords != __DEFAULT_KEYWORDS.get(function, {}):
        raise ValueError(
            f"Operation {getattr(function, '__name__', function)!r} with arguments {keywords} "
            "cannot be serialized"
        )
    if (unary := __UNARY_OPERATORS.get(function)) is not None:
        return UnaryExpression(operator=unary, operand=expression_from_live(node.lhs))
    if (reduction := __REDUCTION_OPERATORS.get(function)) is not None:
        return ReductionExpression(
            operator=reduction, operand=expression_from_live(node.lhs)
        )
    raise TypeError(f"Unsupported column operation: {node.operation!r}")


def __require_rhs(node: Column | DerivedScalarValue) -> ExpressionOperand:
    if node.rhs is None:
        raise TypeError(f"Operation {node.operation!r} requires two operands")
    return node.rhs


def __unwrap(
    operation: Callable[..., object],
) -> tuple[Callable[..., object], dict[str, object]]:
    if isinstance(operation, partial):
        if operation.args:
            raise ValueError(f"Operation {operation!r} cannot be serialized")
        return operation.func, dict(operation.keywords)
    return operation, {}


def __scalar_from_live(
    value: u.Quantity | float | np.integer | np.floating,
) -> NumberLiteral | QuantityLiteral:
    match value:
        case bool() | np.bool_():
            raise TypeError("Boolean values cannot be used as expression literals")
        case u.Quantity():
            if not value.isscalar:
                raise ValueError(
                    f"Only scalar quantities can be serialized, got shape {value.shape}"
                )
            return QuantityLiteral(
                value=value.value.item(), unit=value.unit.to_string()
            )
        case int() | float() | np.integer() | np.floating():
            return NumberLiteral(
                value=value.item() if isinstance(value, np.generic) else value
            )
    raise TypeError(f"Unsupported literal type: {type(value).__name__}")


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


def region_from_live(region: Region) -> RegionMessage:
    """Build a serialized region message from an OpenCosmo region.

    Raises
    ------
    TypeError
        If the region kind has no wire form.
    """
    match region:
        case BoxRegion():
            (x0, x1), (y0, y1), (z0, z1) = region.bounds
            return BoxRegionMessage(
                p1=(float(x0), float(y0), float(z0)),
                p2=(float(x1), float(y1), float(z1)),
            )
        case ConeRegion():
            return ConeRegionMessage(
                center=(float(region.center.ra.deg), float(region.center.dec.deg)),
                radius=float(region.radius.to_value(u.deg)),
            )
        case SkyboxRegion():
            model = region.into_model()
            return SkyboxRegionMessage(
                p1=(float(model.p1[0]), float(model.p1[1])),
                p2=(float(model.p2[0]), float(model.p2[1])),
            )
        case HealpixRegion():
            return HealpixRegionMessage(
                pixels=frozenset(region.pixels.tolist()), nside=int(region.nside)
            )
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


def from_column(column: Column) -> Expression:
    """Build a serialized expression from a lazy column."""
    if not isinstance(column, Column):
        raise TypeError("Expected a column expression")
    return expression_from_live(column)


def from_column_or_scalar(value: Column | DerivedScalarValue) -> Expression:
    """Build a serialized expression from a lazy column or scalar reduction."""
    if not isinstance(value, Column | DerivedScalarValue):
        raise TypeError("Expected a column or a scalar reduction")
    return expression_from_live(value)


def from_unit(unit: u.UnitBase | str) -> str:
    """Build a normalized unit string from an Astropy unit."""
    return u.Unit(unit).to_string()


def from_rows(rows: np.ndarray | DataIndex) -> tuple[int, ...]:
    """Build row positions from a row index.

    Raises
    ------
    TypeError
        If the index is not an integer array.
    """
    array = into_array(rows)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"Row indices must be integers, got {array.dtype}")
    return tuple(array.tolist())


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
    "expression_from_live",
    "expression_to_live",
    "from_column",
    "from_column_or_scalar",
    "from_rows",
    "from_unit",
    "mask_from_live",
    "mask_to_live",
    "region_from_live",
    "region_to_live",
    "selection_to_live",
    "to_column",
    "to_column_or_scalar",
    "to_rows",
    "to_unit",
    "unit_conversions_to_live",
    "unit_settings_to_live",
]
