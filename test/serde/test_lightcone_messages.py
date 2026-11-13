"""Generated Lightcone messages: signature parity, validation, and application."""

import inspect
import json
import os
from pathlib import Path
from unittest.mock import Mock

import astropy.units as u
import numpy as np
import opencosmo as oc
import pytest
from opencosmo.column.column import Column, ColumnMask, DerivedScalarValue
from opencosmo.serde import (
    LightconeMessage,
    LightconeSelectMessage,
    ParameterRole,
    message_signature,
)
from opencosmo.serde.messages import lightcone as messages
from pydantic import TypeAdapter, ValidationError

ADAPTER: TypeAdapter[LightconeMessage] = TypeAdapter(LightconeMessage)
SNAPSHOT = Path(__file__).parent / "snapshots" / "lightcone_messages.schema.json"
MODELS = {model.method: model for model in messages.LIGHTCONE_MESSAGE_TYPES}
METHODS = {
    "filter",
    "select",
    "drop",
    "sort_by",
    "take",
    "take_range",
    "take_rows",
    "bound",
    "with_new_columns",
    "with_units",
    "with_redshift_range",
    "pixel_search",
}

MASS = "fof_halo_mass"
TAG = "fof_halo_tag"
COLUMN = {"kind": "column", "name": MASS}
MASK = {
    "kind": "comparison",
    "operator": "greater_than",
    "left": COLUMN,
    "right": {"kind": "number", "value": 1e13},
}
DOUBLED = {
    "kind": "arithmetic",
    "operator": "multiply",
    "left": COLUMN,
    "right": {"kind": "number", "value": 2.0},
}
MEAN = {"kind": "reduction", "operator": "mean", "operand": COLUMN}
CONE = {"kind": "cone", "center": [45.0, -45.0], "radius": 2.0}

ROLES = {
    inspect.Parameter.POSITIONAL_OR_KEYWORD: ParameterRole.POSITIONAL,
    inspect.Parameter.VAR_POSITIONAL: ParameterRole.VAR_ARGS,
    inspect.Parameter.KEYWORD_ONLY: ParameterRole.KEYWORD_ONLY,
    inspect.Parameter.VAR_KEYWORD: ParameterRole.VAR_KWARGS,
}


@pytest.fixture
def lightcone(test_data):
    return oc.open(
        test_data.lightcone.step(600).halo_properties,
        test_data.lightcone.step(601).halo_properties,
    )


def apply(lightcone, payload):
    return ADAPTER.validate_json(json.dumps(payload)).apply(lightcone)


def assert_same_data(result, expected, *columns):
    data = result.get_data(wrap_single=True)
    expected_data = expected.get_data(wrap_single=True)
    assert data.colnames == expected_data.colnames
    for column in columns:
        assert np.array_equal(data[column], expected_data[column])
        assert data[column].unit == expected_data[column].unit


def test_every_method_has_a_message():
    from opencosmo import serde

    assert set(MODELS) == METHODS
    assert set(messages.__all__) <= set(serde.__all__)
    for name in messages.__all__:
        assert getattr(serde, name) is getattr(messages, name)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_message_mirrors_signature_and_defaults(method):
    model = MODELS[method]
    parameters = list(
        inspect.signature(getattr(oc.Lightcone, method)).parameters.values()
    )[1:]

    assert model.method == method
    assert model.model_fields["kind"].default == method
    assert message_signature(model) == tuple(
        (parameter.name, ROLES[parameter.kind]) for parameter in parameters
    )
    for parameter in parameters:
        field = model.model_fields[parameter.name]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            assert field.get_default(call_default_factory=True) == ()
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            assert field.get_default(call_default_factory=True) == {}
        elif parameter.default is inspect.Parameter.empty:
            assert field.is_required()
        else:
            assert field.get_default() == parameter.default


PAYLOADS = [
    {"kind": "filter", "masks": [MASK], "mode": "local"},
    {"kind": "select", "columns": [TAG, [MASS]], "derived_columns": {"m2": DOUBLED}},
    {"kind": "drop", "columns": [TAG, [MASS]]},
    {"kind": "sort_by", "column": None},
    {"kind": "take", "n": 3, "at": "end", "mode": "global"},
    {"kind": "take_range", "start": 1, "end": 4, "mode": "global"},
    {"kind": "take_rows", "rows": [2, 0, 4]},
    {"kind": "bound", "region": CONE, "select_by": "redshift"},
    {
        "kind": "with_new_columns",
        "descriptions": {"m2": "doubled"},
        "allow_overwrite": True,
        "mode": "local",
        "columns": {"m2": DOUBLED},
    },
    {
        "kind": "with_units",
        "convention": "physical",
        "conversions": {"Mpc": "kpc"},
        "columns": {MASS: "kg"},
    },
    {"kind": "with_redshift_range", "z_low": 0.039, "z_high": 0.04},
    {"kind": "pixel_search", "pixels": [0, 11, 0], "nside": 1},
]


@pytest.mark.parametrize("payload", PAYLOADS, ids=lambda p: p["kind"])
def test_json_round_trip_and_apply_call_parity(payload):
    message = ADAPTER.validate_json(json.dumps(payload))
    assert type(message) is MODELS[payload["kind"]]
    assert ADAPTER.validate_json(ADAPTER.dump_json(message)) == message
    assert ADAPTER.validate_python(message.model_dump(mode="json")) == message

    recorder = Mock(spec=oc.Lightcone)
    sentinel = object()
    getattr(recorder, message.method).return_value = sentinel
    assert message.apply(recorder) is sentinel
    assert len(recorder.mock_calls) == 1
    call = getattr(recorder, message.method).call_args
    match message.method:
        case "filter":
            assert len(call.args) == 1
            assert isinstance(call.args[0], ColumnMask)
            assert call.kwargs == {"mode": "local"}
        case "select":
            assert call.args == (TAG, (MASS,))
            assert call.kwargs["mode"] == "global"
            assert isinstance(call.kwargs["m2"], Column)
            assert call.kwargs["m2"].uuid == (oc.col(MASS) * 2.0).uuid
        case "take_rows" | "pixel_search":
            key = "rows" if message.method == "take_rows" else "pixels"
            np.testing.assert_array_equal(call.args[0], payload[key])
            assert call.args[0].dtype == np.int64
            assert call.args[0].ndim == 1
            assert call.args[1:] == (() if key == "rows" else (1,))
            assert call.kwargs == {}
        case "bound":
            expected = oc.make_cone((45.0, -45.0), 2.0)
            assert type(call.args[0]) is type(expected)
            assert call.args[0].into_model() == expected.into_model()
            assert call.args[1:] == ("redshift",)
        case "with_new_columns":
            assert call.args == ({"m2": "doubled"}, True, "local")
            assert set(call.kwargs) == {"m2"}
            assert call.kwargs["m2"].uuid == (oc.col(MASS) * 2.0).uuid
        case "with_units":
            assert call.args == ("physical", {u.Mpc: u.kpc})
            assert call.kwargs == {MASS: u.kg}
        case _:
            expected_args = {
                "drop": (TAG, (MASS,)),
                "sort_by": (None, False),
                "take": (3, "end", "global"),
                "take_range": (1, 4, "global"),
                "with_redshift_range": (0.039, 0.04),
            }
            assert call.args == expected_args[message.method]
            assert call.kwargs == {}


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"kind": "take", "n": -1}, "greater than or equal to 0"),
        ({"kind": "take", "n": True}, "valid integer"),
        ({"kind": "take", "n": 1, "at": "middle"}, "'start', 'end' or 'random'"),
        ({"kind": "filter", "masks": [MASK], "mode": "everywhere"}, "'local' or"),
        ({"kind": "filter", "masks": [DOUBLED]}, "does not match"),
        ({"kind": "filter", "masks": [MASK], "on_galaxies": True}, "Extra inputs"),
        ({"kind": "take_range", "start": 4, "end": 1}, "end must be greater"),
        ({"kind": "take_rows", "rows": [-1]}, "greater than or equal to 0"),
        ({"kind": "sort_by"}, "Field required"),
        ({"kind": "sort_by", "column": TAG, "invert": "yes"}, "valid boolean"),
        ({"kind": "with_units", "columns": {MASS: "not_a_unit"}}, "Invalid Astropy"),
        ({"kind": "with_units", "convention": "metric"}, "'scalefree'"),
        ({"kind": "with_units", "columns": {"convention": "kg"}}, "Reserved keyword"),
        ({"kind": "select", "derived_columns": {"": MEAN}}, "at least 1 character"),
        ({"kind": "select", "derived_columns": {"mode": MEAN}}, "Reserved keyword"),
        ({"kind": "with_new_columns", "columns": {"mode": DOUBLED}}, "Reserved"),
        (
            {
                "kind": "with_new_columns",
                "descriptions": {"other": "x"},
                "columns": {"m2": DOUBLED},
            },
            "unknown columns",
        ),
        ({"kind": "with_redshift_range", "z_low": float("nan"), "z_high": 1}, "finite"),
        ({"kind": "with_redshift_range", "z_low": 0, "z_high": True}, "valid number"),
        ({"kind": "pixel_search", "pixels": [], "nside": 1}, "Pixels must be"),
        ({"kind": "pixel_search", "pixels": [0], "nside": 3}, "positive power of two"),
        ({"kind": "pixel_search", "pixels": [12], "nside": 1}, "less than 12"),
        ({"kind": "pixel_search", "pixels": [-1]}, "greater than or equal to 0"),
        ({"kind": "drop", "columns": [TAG], "extra": 1}, "Extra inputs"),
        ({"kind": "with_datasets", "datasets": ["a"]}, "does not match any of the"),
    ],
)
def test_messages_reject_invalid_payloads(payload, error):
    with pytest.raises(ValidationError, match=error):
        ADAPTER.validate_python(payload)
    with pytest.raises(ValidationError, match=error):
        ADAPTER.validate_json(json.dumps(payload))


def test_messages_are_immutable_with_independent_defaults():
    first, second = LightconeSelectMessage(), LightconeSelectMessage()

    assert first.derived_columns is not second.derived_columns
    with pytest.raises(ValidationError, match="frozen"):
        first.mode = "local"


def test_select_converts_columns_and_reductions():
    recorder = Mock(spec=oc.Lightcone)
    LightconeSelectMessage.model_validate(
        {"derived_columns": {"m2": DOUBLED, "mean": MEAN}}
    ).apply(recorder)
    derived = recorder.select.call_args.kwargs

    assert isinstance(derived["m2"], Column)
    assert isinstance(derived["mean"], DerivedScalarValue)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "select", "derived_columns": {"one": {"kind": "number", "value": 1}}},
        {"kind": "with_new_columns", "columns": {"m": MEAN}},
    ],
)
def test_apply_rejects_expressions_of_the_wrong_kind(payload):
    recorder = Mock(spec=oc.Lightcone)
    with pytest.raises(TypeError, match="must produce"):
        apply(recorder, payload)
    assert recorder.mock_calls == []


def test_messages_transform_a_lightcone_like_the_api(lightcone):
    payloads = [
        {"kind": "filter", "masks": [MASK]},
        {"kind": "with_new_columns", "columns": {"m2": DOUBLED}},
        {"kind": "select", "columns": [TAG, "m2", MASS]},
        {"kind": "drop", "columns": ["m2"]},
        {"kind": "sort_by", "column": MASS, "invert": True},
        {"kind": "take_range", "start": 0, "end": 20},
        {"kind": "take_rows", "rows": [1, 3, 5, 7]},
        {"kind": "take", "n": 2, "at": "start"},
        {"kind": "with_units", "conversions": {"Msun": "kg"}},
    ]
    result = lightcone
    for payload in payloads:
        result = apply(result, payload)

    expected = (
        lightcone.filter(oc.col(MASS) > 1e13)
        .with_new_columns(m2=oc.col(MASS) * 2.0)
        .select(TAG, "m2", MASS)
        .drop("m2")
        .sort_by(MASS, invert=True)
        .take_range(0, 20)
        .take_rows(np.array([1, 3, 5, 7]))
        .take(2, at="start")
        .with_units(conversions={u.Msun: u.kg})
    )

    assert isinstance(result, oc.Lightcone)
    assert_same_data(result, expected, TAG, MASS)
    assert result.get_data()[MASS].unit == u.kg


def test_lightcone_specific_messages_match_api(lightcone):
    redshift = apply(
        lightcone, {"kind": "with_redshift_range", "z_low": 0.039, "z_high": 0.04}
    )
    pixels = apply(lightcone, {"kind": "pixel_search", "pixels": [0, 1], "nside": 1})
    bound = apply(lightcone, {"kind": "bound", "region": CONE})

    expected_redshift = lightcone.with_redshift_range(0.039, 0.04)
    expected_pixels = lightcone.pixel_search(np.array([0, 1]), nside=1)
    expected_bound = lightcone.bound(oc.make_cone((45.0, -45.0), 2.0))

    assert redshift.z_range == expected_redshift.z_range == (0.039, 0.04)
    assert 0 < len(redshift) == len(expected_redshift) < len(lightcone)
    assert 0 < len(pixels) == len(expected_pixels) < len(lightcone)
    assert len(bound) == len(expected_bound)
    assert_same_data(redshift.select(TAG), expected_redshift.select(TAG), TAG)
    assert_same_data(pixels.select(TAG), expected_pixels.select(TAG), TAG)


def test_column_unit_conversion_matches_api(lightcone):
    result = apply(
        lightcone.select(MASS), {"kind": "with_units", "columns": {MASS: "kg"}}
    )
    expected = lightcone.select(MASS).with_units(fof_halo_mass=u.kg)

    assert result.get_data().unit == u.kg
    assert u.allclose(result.get_data(), expected.get_data())


def test_select_with_reduction_matches_api(lightcone):
    result = apply(lightcone, {"kind": "select", "derived_columns": {"mean": MEAN}})
    expected = lightcone.select(mean=oc.col(MASS).mean())

    assert u.allclose(result.get_data(), expected.get_data())


def test_lightcone_message_schema_matches_snapshot():
    schema = json.dumps(ADAPTER.json_schema(), indent=2, sort_keys=True) + "\n"
    if os.environ.get("OPENCOSMO_UPDATE_SNAPSHOTS"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(schema)
    assert SNAPSHOT.read_text() == schema, (
        "The Lightcone message schema changed. If the API change is intended, rerun "
        "with OPENCOSMO_UPDATE_SNAPSHOTS=1 and review the snapshot diff."
    )
