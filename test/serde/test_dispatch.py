"""Parsing decoded JSON into messages and applying them to OpenCosmo objects."""

import json

import numpy as np
import opencosmo as oc
import pytest
from opencosmo.serde import (
    BoxRegionMessage,
    DatasetTakeMessage,
    MessageError,
    apply_message,
    execute_message,
    parse_message,
)

TAKE = {"kind": "take", "n": 5, "at": "start"}


@pytest.fixture
def halo_properties(test_data):
    return oc.open(test_data.snapshot.primary.halo_properties)


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
