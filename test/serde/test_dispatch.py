"""Parsing decoded JSON into messages and applying them to OpenCosmo objects."""

import json
from unittest.mock import Mock

import numpy as np
import opencosmo as oc
import pytest
from opencosmo.serde import (
    BoxRegionMessage,
    DatasetTakeMessage,
    LightconePixelSearchMessage,
    LightconeTakeMessage,
    MessageError,
    apply_message,
    execute_message,
    parse_message,
)

TAKE = {"kind": "take", "n": 5, "at": "start"}


@pytest.fixture
def halo_properties(test_data):
    return oc.open(test_data.snapshot.primary.halo_properties)


@pytest.fixture
def lightcone(test_data):
    return oc.open(
        test_data.lightcone.step(600).halo_properties,
        test_data.lightcone.step(601).halo_properties,
    )


def test_parse_selects_the_message_for_the_target(halo_properties):
    message = parse_message(type(halo_properties), json.loads(json.dumps(TAKE)))

    assert message == DatasetTakeMessage(n=5, at="start")


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "take", "n": -1}, "greater than or equal to 0"),
        ({"kind": "pixel_search", "pixels": [1]}, "does not match any of the"),
        ({"n": 5}, "Unable to extract tag"),
    ],
)
def test_parse_returns_an_error_for_invalid_payloads(halo_properties, payload, detail):
    error = parse_message(type(halo_properties), payload)

    assert isinstance(error, MessageError)
    assert error.kind == "invalid_message"
    assert error.target == "Dataset"
    assert detail in error.detail


def test_unsupported_targets_return_an_error():
    for result in (parse_message(str, TAKE), execute_message(3, TAKE)):
        assert isinstance(result, MessageError)
        assert result.kind == "unsupported_target"


def test_apply_rejects_messages_for_other_types(halo_properties):
    error = apply_message(halo_properties, BoxRegionMessage(p1=(0, 0, 0), p2=(1, 1, 1)))

    assert isinstance(error, MessageError)
    assert error.kind == "unsupported_message"


def test_execute_parses_and_applies(halo_properties):
    result = execute_message(halo_properties, TAKE)
    expected = halo_properties.take(5, at="start")

    assert isinstance(result, oc.Dataset)
    data, expected_data = result.get_data(), expected.get_data()
    assert data.colnames == expected_data.colnames
    assert all(np.array_equal(data[c], expected_data[c]) for c in data.colnames)


def test_errors_serialize_to_json(halo_properties):
    error = parse_message(type(halo_properties), {"kind": "take", "n": "5"})

    assert MessageError.model_validate_json(error.model_dump_json()) == error


def test_parse_selects_the_lightcone_union(lightcone):
    assert parse_message(oc.Lightcone, TAKE) == LightconeTakeMessage(n=5, at="start")
    pixels = parse_message(oc.Lightcone, {"kind": "pixel_search", "pixels": [1]})
    assert pixels == LightconePixelSearchMessage(pixels=(1,))


def test_lightcone_parse_errors_name_the_target():
    error = parse_message(oc.Lightcone, {"kind": "take", "n": -1})

    assert isinstance(error, MessageError)
    assert error.kind == "invalid_message"
    assert error.target == "Lightcone"


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("halo_properties", LightconeTakeMessage(n=1)),
        ("lightcone", DatasetTakeMessage(n=1)),
    ],
)
def test_apply_rejects_cross_target_messages(request, fixture, message):
    target = request.getfixturevalue(fixture)
    error = apply_message(target, message)

    assert isinstance(error, MessageError)
    assert error.kind == "unsupported_message"
    assert error.target == type(target).__name__
    assert type(message).__name__ in error.detail


def test_execute_applies_lightcone_messages(lightcone):
    result = execute_message(lightcone, TAKE)
    expected = lightcone.take(5, at="start")

    assert isinstance(result, oc.Lightcone)
    data, expected_data = result.get_data(), expected.get_data()
    assert data.colnames == expected_data.colnames
    assert all(np.array_equal(data[c], expected_data[c]) for c in data.colnames)


@pytest.mark.parametrize(
    "target_type", [oc.StructureCollection, oc.HealpixMap, oc.SimulationCollection]
)
def test_collection_targets_without_messages_are_rejected(target_type):
    target = Mock(spec=target_type)
    for result in (
        parse_message(target_type, TAKE),
        apply_message(target, LightconeTakeMessage(n=1)),
        execute_message(target, TAKE),
    ):
        assert isinstance(result, MessageError)
        assert result.kind == "unsupported_target"
    assert target.mock_calls == []
