import astropy.units as u
import numpy as np
import opencosmo as oc
import pytest
from opencosmo.serde import (
    Arctan2Expression,
    ArithmeticExpression,
    ArithmeticOperator,
    BoxRegionMessage,
    ColumnReference,
    ComparisonMask,
    ComparisonOperator,
    CompoundMask,
    ConeRegionMessage,
    HealpixRegionMessage,
    MembershipMask,
    NumberLiteral,
    QuantileExpression,
    QuantityLiteral,
    ReductionExpression,
    ReductionOperator,
    SkyboxRegionMessage,
    UnaryExpression,
    UnaryOperator,
)
from opencosmo.serde.live import (
    expression_from_live,
    expression_to_live,
    from_column,
    from_column_or_scalar,
    from_rows,
    from_unit,
    mask_from_live,
    mask_to_live,
    region_from_live,
    region_to_live,
    to_rows,
    to_unit,
)

MASS = ColumnReference(name="mass")
RADIUS = ColumnReference(name="radius")

EXPRESSIONS = [
    MASS,
    NumberLiteral(value=3),
    NumberLiteral(value=2.5),
    QuantityLiteral(value=1e13, unit="solMass"),
    *(
        ArithmeticExpression(operator=operation, left=MASS, right=RADIUS)
        for operation in ArithmeticOperator
    ),
    ArithmeticExpression(
        operator=ArithmeticOperator.MULTIPLY,
        left=NumberLiteral(value=2),
        right=MASS,
    ),
    *(UnaryExpression(operator=operation, operand=MASS) for operation in UnaryOperator),
    Arctan2Expression(left=MASS, right=RADIUS),
    Arctan2Expression(left=MASS, right=NumberLiteral(value=1.0)),
    *(
        ReductionExpression(operator=operation, operand=MASS)
        for operation in ReductionOperator
    ),
    QuantileExpression(operand=MASS, quantile=0.25),
    ArithmeticExpression(
        operator=ArithmeticOperator.DIVIDE,
        left=ArithmeticExpression(
            operator=ArithmeticOperator.SUBTRACT,
            left=MASS,
            right=ReductionExpression(operator=ReductionOperator.MEAN, operand=MASS),
        ),
        right=ReductionExpression(
            operator=ReductionOperator.STANDARD_DEVIATION, operand=MASS
        ),
    ),
    ArithmeticExpression(
        operator=ArithmeticOperator.ADD,
        left=ReductionExpression(operator=ReductionOperator.MEAN, operand=MASS),
        right=NumberLiteral(value=1),
    ),
]


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_expression_round_trip(expression):
    assert expression_from_live(expression_to_live(expression)) == expression


MASKS = [
    *(
        ComparisonMask(operator=operation, left=MASS, right=NumberLiteral(value=1))
        for operation in ComparisonOperator
    ),
    ComparisonMask(
        operator=ComparisonOperator.LESS_THAN,
        left=MASS,
        right=ReductionExpression(operator=ReductionOperator.MEDIAN, operand=MASS),
    ),
    MembershipMask(
        operand=MASS,
        values=(NumberLiteral(value=1), QuantityLiteral(value=2.0, unit="Mpc")),
    ),
    CompoundMask(
        operator="and",
        left=ComparisonMask(
            operator=ComparisonOperator.GREATER_THAN,
            left=MASS,
            right=QuantityLiteral(value=1e13, unit="solMass"),
        ),
        right=CompoundMask(
            operator="or",
            left=MembershipMask(operand=RADIUS, values=(NumberLiteral(value=4),)),
            right=ComparisonMask(
                operator=ComparisonOperator.EQUAL, left=RADIUS, right=MASS
            ),
        ),
    ),
]


@pytest.mark.parametrize("mask", MASKS)
def test_mask_round_trip(mask):
    assert mask_from_live(mask_to_live(mask)) == mask


def test_user_built_expression_from_live():
    mass = oc.col("mass")
    expression = (mass - mass.mean()) / mass.std() + np.float64(1.5) * u.Mpc

    assert expression_from_live(expression) == ArithmeticExpression(
        operator=ArithmeticOperator.ADD,
        left=EXPRESSIONS[-2],
        right=QuantityLiteral(value=1.5, unit="Mpc"),
    )


def test_user_built_mask_from_live():
    mask = (oc.col("mass") > 5) & oc.col("radius").isin(np.array([1, 2]))

    assert mask_from_live(mask) == CompoundMask(
        operator="and",
        left=ComparisonMask(
            operator=ComparisonOperator.GREATER_THAN,
            left=MASS,
            right=NumberLiteral(value=5),
        ),
        right=MembershipMask(
            operand=RADIUS, values=(NumberLiteral(value=1), NumberLiteral(value=2))
        ),
    )


def test_expression_from_live_rejects_custom_log_container():
    with pytest.raises(ValueError, match="cannot be serialized"):
        expression_from_live(oc.col("mass").log10(u.MagUnit))


def test_expression_from_live_rejects_vector_quantity():
    with pytest.raises(ValueError, match="Only scalar quantities"):
        mask_from_live(oc.col("mass") > [1, 2] * u.Mpc)


def test_expression_from_live_rejects_booleans():
    with pytest.raises(TypeError, match="Boolean"):
        mask_from_live(oc.col("mass") == True)


def test_from_column_rejects_scalar_reduction():
    with pytest.raises(TypeError, match="Expected a column"):
        from_column(oc.col("mass").mean())
    assert from_column_or_scalar(oc.col("mass").mean()) == ReductionExpression(
        operator=ReductionOperator.MEAN, operand=MASS
    )


REGIONS = [
    BoxRegionMessage(p1=(0.0, 1.0, 2.0), p2=(10.0, 11.0, 12.0)),
    ConeRegionMessage(center=(45.0, -30.0), radius=2.5),
    SkyboxRegionMessage(p1=(350.0, -10.0), p2=(10.0, 10.0)),
    HealpixRegionMessage(pixels=frozenset({3, 1, 7}), nside=4),
]


@pytest.mark.parametrize("region", REGIONS)
def test_region_round_trip(region):
    restored = region_from_live(region_to_live(region))

    assert restored.model_dump() == pytest.approx(region.model_dump())


def test_cone_radius_from_live_is_degrees():
    region = oc.make_cone((10, 20), 30 * u.arcmin)

    assert region_from_live(region).radius == pytest.approx(0.5)


def test_rows_and_units_round_trip():
    rows = (0, 4, 9)
    assert from_rows(to_rows(rows)) == rows
    assert from_rows((np.array([2, 10]), np.array([2, 1]))) == (2, 3, 10)
    assert from_unit(to_unit("M_sun")) == "solMass"
