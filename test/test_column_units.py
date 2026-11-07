"""Pure unit tests for column unit inference and producer unit metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import astropy.units as u
import numpy as np
import pytest
from opencosmo.column.column import (
    Column,
    ConstructedColumn,
    DerivedScalarValue,
    EvaluatedColumn,
    RawColumn,
    col,
)
from opencosmo.column.evaluate import EvaluateStrategy
from opencosmo.dataset.graph import get_derived_units
from opencosmo.units import UnitsError

if TYPE_CHECKING:
    from opencosmo.units.types import UnitValue

# ---------------------------------------------------------------------------
# Expression inference (unbound, unnamed)
# ---------------------------------------------------------------------------


class TestInferUnit:
    def test_bare_column_returns_lookup(self):
        assert col("x").infer_unit({"x": u.m}) == u.m

    def test_product_and_ratio(self):
        units = {"x": u.m, "y": u.s}
        assert (col("x") * col("y")).infer_unit(units) == u.m * u.s
        assert (col("x") / col("y")).infer_unit(units) == u.m / u.s

    def test_power(self):
        assert (col("x") ** 2).infer_unit({"x": u.m}) == u.m**2

    def test_unitless_inputs(self):
        assert (col("x") * col("y")).infer_unit({"x": None, "y": None}) is None

    def test_one_sided_unit(self):
        assert (col("x") * col("y")).infer_unit({"x": u.m, "y": None}) == u.m
        assert (col("x") * col("y")).infer_unit({"x": None, "y": u.m}) == u.m

    def test_unit_quantity_operand(self):
        result = (col("x") * (3 * u.s)).infer_unit({"x": u.m})
        assert result == u.m * u.s

    def test_scalar_operand_preserves_quantity_result(self):
        # Existing behaviour: scalar operands yield a Quantity rather than a unit.
        result = (col("x") * 2).infer_unit({"x": u.m})
        assert isinstance(result, u.Quantity)
        assert result.unit == u.m

    def test_logarithmic_unit(self):
        result = col("x").log10().infer_unit({"x": u.m})
        assert isinstance(result, u.DexUnit)
        assert result.physical_unit == u.m

    def test_missing_input_unit_raises(self):
        with pytest.raises(KeyError, match="x"):
            col("x").infer_unit({})

    def test_nested_expression(self):
        expr = (col("x") * col("y")) / col("z")
        assert expr.infer_unit({"x": u.m, "y": u.s, "z": u.kg}) == u.m * u.s / u.kg

    def test_add_same_units(self):
        assert (col("x") + col("y")).infer_unit({"x": u.m, "y": u.m}) == u.m

    def test_add_mismatched_units_raises(self):
        with pytest.raises(UnitsError):
            (col("x") + col("y")).infer_unit({"x": u.m, "y": u.s})

    def test_subtract_mismatched_units_raises(self):
        with pytest.raises(UnitsError):
            (col("x") - col("y")).infer_unit({"x": u.m, "y": u.s})

    @pytest.mark.parametrize("reducer", ["mean", "std", "min", "max", "median", "sum"])
    def test_scalar_reduction_keeps_unit(self, reducer: str):
        scalar = getattr(col("x"), reducer)()
        assert isinstance(scalar, DerivedScalarValue)
        assert scalar.infer_unit({"x": u.m}) == u.m

    def test_quantile_keeps_unit(self):
        assert col("x").quantile(0.5).infer_unit({"x": u.m}) == u.m

    def test_variance_squares_unit(self):
        assert col("x").var().infer_unit({"x": u.m}) == u.m**2

    def test_normalization_is_dimensionless(self):
        expr = (col("x") - col("x").mean()) / col("x").std()
        assert expr.infer_unit({"x": u.m}) == u.dimensionless_unscaled

    def test_scalar_add_mismatched_raises(self):
        expr = col("x") + col("y").mean()
        with pytest.raises(UnitsError):
            expr.infer_unit({"x": u.m, "y": u.s})

    def test_unnamed_expressions_can_infer(self):
        assert col("x").name is None
        assert col("x").mean().name is None
        assert (col("x") * 2).name is None
        assert col("x").mean().infer_unit({"x": u.m}) == u.m


# ---------------------------------------------------------------------------
# Producer metadata
# ---------------------------------------------------------------------------


def _named(expr: Column | DerivedScalarValue, name: str) -> Column | DerivedScalarValue:
    expr.name = name
    return expr


class TestProducerGetUnits:
    def test_named_column(self):
        expr = _named(col("x") * col("y"), "out")
        assert expr.get_units({"x": u.m, "y": u.s}) == {"out": u.m * u.s}

    def test_named_column_unitless(self):
        expr = _named(col("x") * col("y"), "out")
        assert expr.get_units({"x": None, "y": None}) == {"out": None}

    def test_named_scalar(self):
        expr = _named(col("x").mean(), "avg")
        assert expr.get_units({"x": u.m}) == {"avg": u.m}

    def test_named_scalar_unitless(self):
        expr = _named(col("x").mean(), "avg")
        assert expr.get_units({"x": None}) == {"avg": None}

    def test_unnamed_column_raises(self):
        with pytest.raises(
            RuntimeError, match="Cannot get output units for an unnamed expression."
        ):
            (col("x") * col("y")).get_units({"x": u.m, "y": u.s})

    def test_unnamed_scalar_raises(self):
        with pytest.raises(
            RuntimeError, match="Cannot get output units for an unnamed expression."
        ):
            col("x").mean().get_units({"x": u.m})


# ---------------------------------------------------------------------------
# RawColumn
# ---------------------------------------------------------------------------


class TestRawColumnGetUnits:
    def test_lookup(self):
        rc = RawColumn("x", "desc", UUID(int=1))
        assert rc.get_units({"x": u.m, "y": u.s}) == {"x": u.m}

    def test_lookup_unitless(self):
        rc = RawColumn("x", "desc", UUID(int=1))
        assert rc.get_units({"x": None}) == {"x": None}

    def test_alias_reports_under_output_name(self):
        rc = RawColumn("x", "desc", UUID(int=1), alias="renamed")
        assert rc.get_units({"x": u.kg}) == {"renamed": u.kg}


# ---------------------------------------------------------------------------
# EvaluatedColumn
# ---------------------------------------------------------------------------


def _evaluated(units: dict[str, UnitValue]) -> EvaluatedColumn:
    def func(**kwargs: np.ndarray) -> np.ndarray:
        return np.zeros(1)

    return EvaluatedColumn(
        func,
        {"x"},
        set(units),
        "numpy",
        units,
        EvaluateStrategy.ROW_WISE,
    )


class TestEvaluatedColumnGetUnits:
    def test_multiple_outputs_including_none(self):
        ec = _evaluated({"a": u.m, "b": None})
        assert ec.get_units({"x": u.s}) == {"a": u.m, "b": None}

    def test_ignores_input_units(self):
        ec = _evaluated({"a": u.m})
        assert ec.get_units({}) == ec.get_units({"x": u.kg})

    def test_returned_mapping_is_a_copy(self):
        ec = _evaluated({"a": u.m})
        ec.get_units({})["a"] = u.s
        assert ec.get_units({}) == {"a": u.m}

    def test_constructor_copies_units(self):
        units: dict[str, UnitValue] = {"a": u.m}
        ec = _evaluated(units)
        units["a"] = u.s
        assert ec.get_units({}) == {"a": u.m}


# ---------------------------------------------------------------------------
# Graph-level metadata
# ---------------------------------------------------------------------------


class TestGetDerivedUnits:
    def test_dependency_chain_skips_raw(self):
        raw_x = RawColumn("x", "x", UUID(int=1))
        raw_y = RawColumn("y", "y", UUID(int=2))
        name_to_uuid = {"x": raw_x.uuid, "y": raw_y.uuid}
        known = set(name_to_uuid.values())

        speed = _named(col("x") / col("y"), "speed").bind(name_to_uuid, known)
        assert speed.uuid is not None
        name_to_uuid["speed"] = speed.uuid
        known.add(speed.uuid)

        # Depends on a derived column, so the units must flow through the chain.
        accel = _named(col("speed") / col("y"), "accel").bind(name_to_uuid, known)
        assert accel.uuid is not None
        known.add(accel.uuid)

        producers: dict[UUID, ConstructedColumn | RawColumn] = {
            accel.uuid: accel,
            raw_x.uuid: raw_x,
            speed.uuid: speed,
            raw_y.uuid: raw_y,
        }
        result = get_derived_units(producers, {"x": u.m, "y": u.s})

        assert result == {"speed": u.m / u.s, "accel": u.m / u.s**2}

    def test_multiple_output_producer(self):
        raw_x = RawColumn("x", "x", UUID(int=1))
        name_to_uuid = {"x": raw_x.uuid}
        known = set(name_to_uuid.values())
        ec = _evaluated({"a": u.m, "b": None}).bind(name_to_uuid, known)
        assert ec.uuid is not None

        result = get_derived_units({raw_x.uuid: raw_x, ec.uuid: ec}, {"x": u.s})
        assert result == {"a": u.m, "b": None}

    def test_input_units_not_mutated(self):
        raw_x = RawColumn("x", "x", UUID(int=1))
        name_to_uuid = {"x": raw_x.uuid}
        known = set(name_to_uuid.values())
        double = _named(col("x") * col("x"), "x2").bind(name_to_uuid, known)
        assert double.uuid is not None
        units: dict[str, UnitValue] = {"x": u.m}

        get_derived_units({raw_x.uuid: raw_x, double.uuid: double}, units)
        assert units == {"x": u.m}
