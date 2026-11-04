import json

import pytest
from opencosmo.serde import (
    ArithmeticExpression,
    ArithmeticOperator,
    ColumnReference,
    ComparisonMask,
    ComparisonOperator,
    CompoundMask,
    Expression,
    Mask,
    MembershipMask,
    NumberLiteral,
    QuantileExpression,
    QuantityLiteral,
    ReductionExpression,
    ReductionOperator,
)
from pydantic import TypeAdapter, ValidationError


def test_nested_expression_round_trip():
    expression = ArithmeticExpression(
        operator=ArithmeticOperator.DIVIDE,
        left=ArithmeticExpression(
            operator=ArithmeticOperator.SUBTRACT,
            left=ColumnReference(name="mass"),
            right=ReductionExpression(
                operator=ReductionOperator.MEAN,
                operand=ColumnReference(name="mass"),
            ),
        ),
        right=ReductionExpression(
            operator=ReductionOperator.STANDARD_DEVIATION,
            operand=ColumnReference(name="mass"),
        ),
    )

    adapter = TypeAdapter(Expression)
    restored = adapter.validate_json(adapter.dump_json(expression))

    assert restored == expression


def test_nested_mask_round_trip():
    mask = CompoundMask(
        operator="and",
        left=ComparisonMask(
            operator=ComparisonOperator.GREATER_THAN,
            left=ColumnReference(name="mass"),
            right=QuantityLiteral(value=1e13, unit="Msun"),
        ),
        right=MembershipMask(
            operand=ColumnReference(name="tag"),
            values=(NumberLiteral(value=1), NumberLiteral(value=4)),
        ),
    )

    adapter = TypeAdapter(Mask)
    restored = adapter.validate_json(adapter.dump_json(mask))

    assert restored == mask


def test_quantity_normalizes_valid_astropy_unit():
    quantity = QuantityLiteral(value=2.0, unit="M_sun")

    assert quantity.unit == "solMass"
    assert json.loads(quantity.model_dump_json()) == {
        "kind": "quantity",
        "value": 2.0,
        "unit": "solMass",
    }


@pytest.mark.parametrize("unit", ["", "definitely_not_a_unit"])
def test_quantity_rejects_invalid_astropy_unit(unit):
    with pytest.raises(
        ValidationError, match="Invalid Astropy unit|at least 1 character"
    ):
        QuantityLiteral(value=2.0, unit=unit)


@pytest.mark.parametrize("value", [True, False, "1", float("nan"), float("inf")])
def test_number_rejects_non_finite_or_non_strict_values(value):
    with pytest.raises(ValidationError):
        NumberLiteral(value=value)


@pytest.mark.parametrize("quantile", [-0.1, 1.1, float("nan"), float("inf")])
def test_quantile_rejects_invalid_values(quantile):
    with pytest.raises(ValidationError):
        QuantileExpression(operand=ColumnReference(name="mass"), quantile=quantile)


def test_quantile_rejects_boolean():
    with pytest.raises(ValidationError):
        QuantileExpression(operand=ColumnReference(name="mass"), quantile=True)


def test_expression_rejects_unknown_operator():
    with pytest.raises(ValidationError):
        ArithmeticExpression(
            operator="modulo",
            left=ColumnReference(name="mass"),
            right=NumberLiteral(value=2),
        )


def test_expression_models_forbid_extra_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ColumnReference(name="mass", uuid="internal")


def test_expression_and_mask_unions_are_distinct():
    comparison = {
        "kind": "comparison",
        "operator": "equal",
        "left": {"kind": "column", "name": "tag"},
        "right": {"kind": "number", "value": 1},
    }

    assert isinstance(TypeAdapter(Mask).validate_python(comparison), ComparisonMask)
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        TypeAdapter(Expression).validate_python(comparison)
