"""Generated Dataset messages: signature parity, validation, and application."""

import inspect
import json
import os
from pathlib import Path

import astropy.units as u
import numpy as np
import opencosmo as oc
import pytest
from opencosmo.column.column import Column, DerivedScalarValue
from opencosmo.dataset import Dataset
from opencosmo.serde import (
    DatasetMessage,
    DatasetSelectMessage,
    ParameterRole,
    message_signature,
)
from opencosmo.serde.messages import dataset as messages
from pydantic import TypeAdapter, ValidationError

ADAPTER: TypeAdapter[DatasetMessage] = TypeAdapter(DatasetMessage)
SNAPSHOT = Path(__file__).parent / "snapshots" / "dataset_messages.schema.json"
MODELS = {model.method: model for model in messages.DATASET_MESSAGE_TYPES}

MASS = "fof_halo_mass"
TAG = "fof_halo_tag"
COLUMN = {"kind": "column", "name": MASS}
MASK = {
    "kind": "comparison",
    "operator": "greater_than",
    "left": COLUMN,
    "right": {"kind": "number", "value": 1e12},
}
DOUBLED = {
    "kind": "arithmetic",
    "operator": "multiply",
    "left": COLUMN,
    "right": {"kind": "number", "value": 2.0},
}
MEAN = {"kind": "reduction", "operator": "mean", "operand": COLUMN}

ROLES = {
    inspect.Parameter.POSITIONAL_OR_KEYWORD: ParameterRole.POSITIONAL,
    inspect.Parameter.VAR_POSITIONAL: ParameterRole.VAR_ARGS,
    inspect.Parameter.KEYWORD_ONLY: ParameterRole.KEYWORD_ONLY,
    inspect.Parameter.VAR_KEYWORD: ParameterRole.VAR_KWARGS,
}


@pytest.fixture
def halo_properties(test_data):
    return oc.open(test_data.snapshot.primary.halo_properties)


def apply(dataset, payload):
    return ADAPTER.validate_json(json.dumps(payload)).apply(dataset)


@pytest.mark.parametrize("method", sorted(MODELS))
def test_message_mirrors_signature(method):
    model = MODELS[method]
    parameters = list(inspect.signature(getattr(Dataset, method)).parameters.values())

    assert model.method == method
    assert message_signature(model) == tuple(
        (parameter.name, ROLES[parameter.kind]) for parameter in parameters[1:]
    )


def test_package_exports_every_message():
    from opencosmo import serde

    assert set(messages.__all__) <= set(serde.__all__)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "filter", "masks": [MASK], "mode": "local"},
        {"kind": "select", "columns": [TAG, [MASS]], "derived_columns": {"m": MEAN}},
        {"kind": "drop", "columns": [TAG]},
        {"kind": "sort_by", "column": None},
        {"kind": "take", "n": 3, "at": "start"},
        {"kind": "take_range", "start": 1, "end": 4},
        {"kind": "take_rows", "rows": [0, 2]},
        {
            "kind": "bound",
            "region": {"kind": "box", "p1": [0, 0, 0], "p2": [10, 10, 10]},
        },
        {
            "kind": "with_new_columns",
            "descriptions": {"m2": "doubled"},
            "new_columns": {"m2": DOUBLED},
        },
        {
            "kind": "with_units",
            "convention": "physical",
            "conversions": {"Mpc": "kpc"},
            "columns": {MASS: "kg"},
        },
    ],
)
def test_messages_round_trip(payload):
    message = ADAPTER.validate_python(payload)

    assert type(message) is MODELS[payload["kind"]]
    assert ADAPTER.validate_json(message.model_dump_json()) == message


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"kind": "take", "n": -1}, "greater than or equal to 0"),
        ({"kind": "take", "n": "3"}, "valid integer"),
        ({"kind": "take", "n": 1, "at": "middle"}, "'start', 'end' or 'random'"),
        ({"kind": "filter", "masks": [MASK], "mode": "everywhere"}, "'local' or"),
        ({"kind": "take_range", "start": 4, "end": 1}, "end must be greater"),
        ({"kind": "take_rows", "rows": [-1]}, "greater than or equal to 0"),
        ({"kind": "sort_by"}, "Field required"),
        ({"kind": "sort_by", "column": TAG, "invert": "yes"}, "valid boolean"),
        ({"kind": "with_units", "columns": {MASS: "not_a_unit"}}, "Invalid Astropy"),
        ({"kind": "with_units", "convention": "metric"}, "'scalefree'"),
        ({"kind": "select", "derived_columns": {"": MEAN}}, "at least 1 character"),
        (
            {
                "kind": "with_new_columns",
                "descriptions": {"other": "x"},
                "new_columns": {"m2": DOUBLED},
            },
            "unknown columns",
        ),
        ({"kind": "drop", "columns": [TAG], "extra": 1}, "Extra inputs"),
        ({"kind": "pixel_search", "pixels": [1]}, "does not match any of the"),
    ],
)
def test_messages_reject_invalid_payloads(payload, error):
    with pytest.raises(ValidationError, match=error):
        ADAPTER.validate_python(payload)


def test_unit_strings_are_normalized():
    message = ADAPTER.validate_python(
        {"kind": "with_units", "conversions": {"Mpc": "kpc"}, "columns": {MASS: "kg"}}
    )

    assert message.conversions == {u.Mpc.to_string(): u.kpc.to_string()}
    assert message.columns == {MASS: u.kg.to_string()}


def test_messages_are_immutable():
    message = ADAPTER.validate_python({"kind": "take", "n": 3})

    with pytest.raises(ValidationError, match="frozen"):
        message.n = 4


def test_select_converts_columns_and_reductions():
    recorded = {}

    class Recorder:
        def select(self, *columns, mode="global", **derived):
            recorded.update(derived)
            return self

    DatasetSelectMessage.model_validate(
        {"derived_columns": {"m2": DOUBLED, "mean": MEAN}}
    ).apply(Recorder())

    assert isinstance(recorded["m2"], Column)
    assert isinstance(recorded["mean"], DerivedScalarValue)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "select", "derived_columns": {"one": {"kind": "number", "value": 1}}},
        {"kind": "with_new_columns", "new_columns": {"m": MEAN}},
    ],
)
def test_apply_rejects_expressions_of_the_wrong_kind(halo_properties, payload):
    with pytest.raises(TypeError, match="must produce"):
        apply(halo_properties, payload)


def test_messages_transform_a_dataset_like_the_api(halo_properties):
    payloads = [
        {"kind": "filter", "masks": [MASK]},
        {"kind": "with_new_columns", "new_columns": {"m2": DOUBLED}},
        {"kind": "select", "columns": [TAG, "m2", MASS]},
        {"kind": "drop", "columns": ["m2"]},
        {"kind": "sort_by", "column": TAG, "invert": True},
        {"kind": "take_range", "start": 0, "end": 20},
        {"kind": "take_rows", "rows": [1, 3, 5, 7]},
        {"kind": "take", "n": 2, "at": "start"},
        {"kind": "with_units", "conversions": {"Msun": "kg"}},
    ]
    result = halo_properties
    for payload in payloads:
        result = apply(result, payload)

    expected = (
        halo_properties.filter(oc.col(MASS) > 1e12)
        .with_new_columns(m2=oc.col(MASS) * 2.0)
        .select(TAG, "m2", MASS)
        .drop("m2")
        .sort_by(TAG, invert=True)
        .take_range(0, 20)
        .take_rows(np.array([1, 3, 5, 7]))
        .take(2, at="start")
        .with_units(conversions={u.Msun: u.kg})
        .get_data()
    )
    data = result.get_data()

    assert data.colnames == expected.colnames
    assert np.array_equal(data[TAG], expected[TAG])
    assert np.array_equal(data[MASS], expected[MASS])
    assert data[MASS].unit == expected[MASS].unit


def test_column_unit_conversion_matches_api(halo_properties):
    result = apply(halo_properties, {"kind": "with_units", "columns": {MASS: "kg"}})
    expected = halo_properties.with_units(fof_halo_mass=u.kg)

    assert result.get_data()[MASS].unit == u.kg
    assert np.allclose(result.get_data()[MASS], expected.get_data()[MASS])


def test_bound_matches_api(halo_properties):
    result = apply(
        halo_properties,
        {"kind": "bound", "region": {"kind": "box", "p1": [0, 0, 0], "p2": [20] * 3}},
    )
    expected = halo_properties.bound(oc.make_box((0, 0, 0), (20, 20, 20)))

    assert len(result) == len(expected) > 0


def test_select_with_reduction_matches_api(halo_properties):
    result = apply(
        halo_properties,
        {"kind": "select", "derived_columns": {"mean": MEAN}},
    )
    expected = halo_properties.select(mean=oc.col(MASS).mean())

    assert u.allclose(result.get_data(), expected.get_data())


def test_dataset_message_schema_matches_snapshot():
    schema = json.dumps(ADAPTER.json_schema(), indent=2, sort_keys=True) + "\n"
    if os.environ.get("OPENCOSMO_UPDATE_SNAPSHOTS"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(schema)
    assert SNAPSHOT.read_text() == schema, (
        "The Dataset message schema changed. If the API change is intended, rerun "
        "with OPENCOSMO_UPDATE_SNAPSHOTS=1 and review the snapshot diff."
    )
