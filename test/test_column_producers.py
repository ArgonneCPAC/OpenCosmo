"""Pure tests for producer binding and non-mutating registration."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import pytest
from opencosmo.collection.lightcone.scope import LightconeScope, partition_columns
from opencosmo.column.column import (
    Column,
    DerivedScalarValue,
    EvaluatedColumn,
    RawColumn,
    col,
)
from opencosmo.column.evaluate import EvaluateStrategy
from opencosmo.column.reducer import LocalReducer
from opencosmo.dataset.columns import __categorize_columns
from opencosmo.dataset.graph import evaluate_producers

if TYPE_CHECKING:
    from opencosmo.column.column import ConstructedColumn


@pytest.mark.parametrize("scalar", [False, True])
def test_expression_binding_is_non_mutating(scalar: bool) -> None:
    expression = col("x").mean() if scalar else col("x") * 2
    reducer = LocalReducer()
    expression = expression.with_reducer(reducer)
    mapping = {"x": UUID(int=1)}
    producer: ConstructedColumn = expression
    first = producer.bind(mapping, set(mapping.values()), output_name="first")
    second = producer.bind(
        mapping, set(mapping.values()), output_name="second", description="new"
    )

    assert expression.name is None
    assert expression.description == "None"
    assert expression.uuid is None
    assert expression.dep_map is None
    assert first.produces == {"first"}
    assert second.produces == {"second"}
    assert first.uuid != second.uuid
    assert first.dep_map == second.dep_map == mapping
    assert first.description == "None"
    assert second.description == "new"
    if isinstance(first, DerivedScalarValue):
        assert first.reducer is reducer
        assert first.no_cache
    data = {"x": np.array([1.0, 3.0])}
    result = first.evaluate(data, index=None)
    assert not isinstance(result, dict)
    np.testing.assert_allclose(result, np.array(2 if scalar else [2, 6]))


@pytest.mark.parametrize("scalar", [False, True])
def test_unnamed_expression_metadata_requires_naming(scalar: bool) -> None:
    expression = col("x").mean() if scalar else col("x")
    with pytest.raises(RuntimeError, match="Cannot get output names for an unnamed"):
        _ = expression.produces
    with pytest.raises(RuntimeError, match="Cannot bind an unnamed"):
        expression.bind({}, set())
    with pytest.raises(RuntimeError, match="has not been bound yet"):
        _ = expression.requires


def test_binding_preserves_existing_metadata_and_cache_policy() -> None:
    expression = Column("x", None, lambda x, _: x, "original", "out", no_cache=True)
    bound = expression.with_reducer(LocalReducer()).bind({"x": UUID(int=1)}, set())
    assert bound.produces == {"out"}
    assert bound.description == "original"
    assert bound.no_cache


def test_binding_rejects_unknown_dependencies() -> None:
    for expression in (col("missing"), col("missing").mean()):
        with pytest.raises(ValueError, match="depends on unknown columns"):
            expression.bind({}, set(), output_name="out")


def test_dataset_registration_reuses_expression_without_mutation() -> None:
    expression = col("x") * 2
    mapping = {"x": UUID(int=1)}
    producers, _, _, names, _ = __categorize_columns(
        {"first": expression, "second": expression},
        mapping,
        set(mapping.values()),
        {"first": "first description", "second": "second description"},
        2,
    )
    assert names == ["first", "second"]
    assert expression.name is None
    assert expression.description == "None"
    assert expression.uuid is None
    assert producers[mapping["first"]].description == "first description"
    assert producers[mapping["second"]].description == "second description"
    raw = RawColumn("x", None, mapping["x"])
    output = evaluate_producers(producers | {raw.uuid: raw}, {"x": np.array([1, 3])})
    np.testing.assert_array_equal(output["first"], [2, 6])
    np.testing.assert_array_equal(output["second"], [2, 6])


def __evaluated() -> EvaluatedColumn:
    def transform(avg: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
        return {"centered": x - avg, "shifted": x + avg}

    return EvaluatedColumn(
        transform,
        {"avg", "x"},
        {"centered", "shifted"},
        "numpy",
        {"centered": None, "shifted": None},
        EvaluateStrategy.VECTORIZE,
        description="original",
        no_cache=True,
    )


def test_evaluated_binding_preserves_declared_outputs_and_input_metadata() -> None:
    expression = __evaluated()
    producer: ConstructedColumn = expression
    bound = producer.bind(
        {"avg": UUID(int=1), "x": UUID(int=2)},
        set(),
        output_name="registration_key",
        description="new",
    )
    assert bound.produces == {"centered", "shifted"}
    assert bound.description == "new"
    assert bound.no_cache
    assert expression.description == "original"
    assert expression.uuid is None
    assert expression.dep_map is None


def test_dataset_registration_does_not_mutate_evaluated_description() -> None:
    expression = __evaluated()
    mapping = {"avg": UUID(int=1), "x": UUID(int=2)}
    producers, _, _, names, _ = __categorize_columns(
        {"registration_key": expression},
        mapping,
        set(mapping.values()),
        {"registration_key": "new"},
        2,
    )
    assert set(names) == {"centered", "shifted"}
    assert len(producers) == 1
    assert next(iter(producers.values())).description == "new"
    assert expression.description == "original"
    assert expression.uuid is None


def test_lightcone_scope_registration_reuses_expression() -> None:
    expression = col("x").mean()
    original = LightconeScope()
    first = original.add({"first": expression}, {"x"})
    second = original.add({"second": expression}, {"x"})
    together = original.add({"first": expression, "second": expression}, {"x"})
    assert not original.names()
    assert expression.name is None
    assert expression.uuid is None
    assert expression.dep_map is None
    assert first.names() == {"first"}
    assert second.names() == {"second"}
    assert together.names() == {"first", "second"}
    assert len(together.placeholders) == 1
    assert len(together.derived_producers) == 2
    assert all(not isinstance(p, RawColumn) for p in together.derived_producers)
    output = evaluate_producers(together.producers, {"x": np.array([1.0, 3.0])})
    assert output == {"first": 2.0, "second": 2.0}


def test_scope_routes_evaluated_producer_and_binds_all_output_names() -> None:
    scope = LightconeScope().add({"avg": col("x").mean()}, {"x"})
    expression = __evaluated()
    scoped, child = partition_columns({"registration_key": expression}, {"x"}, scope)
    assert scoped["registration_key"] is expression
    assert not child
    updated = scope.add(scoped, {"x"})
    assert updated.names() == {"avg", "centered", "shifted"}
    assert updated.required_child_columns() == {"x"}
    assert expression.uuid is None
    assert expression.description == "original"
    output = evaluate_producers(updated.producers, {"x": np.array([1.0, 3.0])})
    np.testing.assert_allclose(output["centered"], [-1, 1])
    np.testing.assert_allclose(output["shifted"], [3, 5])


def test_scope_routes_and_binds_intra_call_dependencies() -> None:
    scope = LightconeScope().add(
        {"avg": col("x").mean(), "centered": col("x") - col("avg")}, {"x"}
    )
    assert scope.names() == {"avg", "centered"}
    assert scope.required_child_columns() == {"x"}
    output = evaluate_producers(scope.producers, {"x": np.array([1.0, 3.0])})
    assert output["avg"] == 2
    np.testing.assert_allclose(output["centered"], [-1, 1])


def test_scope_adds_evaluated_child_dependencies_and_routes_output_names() -> None:
    scope = LightconeScope().add({"avg": col("y").mean()}, {"y", "x"})
    updated = scope.add(
        {
            "registration_key": __evaluated(),
            "double": col("centered") * 2,
        },
        {"y", "x"},
    )
    assert updated.names() == {"avg", "centered", "shifted", "double"}
    assert updated.required_child_columns() == {"x", "y"}
    assert scope.required_child_columns() == {"y"}
    output = evaluate_producers(
        updated.producers, {"x": np.array([1.0, 3.0]), "y": np.array([0.0, 2.0])}
    )
    np.testing.assert_allclose(output["double"], [0, 4])
