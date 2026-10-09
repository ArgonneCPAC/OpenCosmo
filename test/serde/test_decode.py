"""Tests for decoding incoming payloads into messages."""

import json

import pytest
from opencosmo.serde import (
    DESCRIPTORS,
    SerdeError,
    SerdeErrorCategory,
    TakeMessage,
    decode_message,
)
from pydantic import TypeAdapter


def test_decodes_dict_str_and_bytes():
    payload = {"kind": "take", "n": 5, "at": "start"}
    expected = TakeMessage(n=5, at="start")
    assert decode_message("Dataset", payload) == expected
    assert decode_message("Dataset", json.dumps(payload)) == expected
    assert decode_message("Dataset", json.dumps(payload).encode()) == expected


def test_round_trips_every_default_constructible_message():
    # A message whose model validates from {} just from its kind.
    for name, descriptor in DESCRIPTORS.items():
        for kind, model in descriptor.messages_by_kind.items():
            try:
                message = model.model_validate({})
            except Exception:
                continue
            decoded = decode_message(name, message.model_dump_json())
            assert decoded == message, kind


def test_nested_payload_decodes():
    result = decode_message(
        "Dataset",
        {
            "kind": "filter",
            "masks": [
                {
                    "kind": "comparison",
                    "left": {"kind": "column", "name": "mass"},
                    "operator": "greater_than",
                    "right": {"kind": "number", "value": 1},
                }
            ],
        },
    )
    assert not isinstance(result, SerdeError)
    assert result.kind == "filter"


def test_kind_not_allowed_for_target():
    result = decode_message("Dataset", {"kind": "match", "dataset": "a"})
    assert isinstance(result, SerdeError)
    assert result.operation == "decode_message"
    assert result.target_type == "Dataset"
    assert "not supported for Dataset" in result.message


def test_same_kind_decodes_for_other_target():
    result = decode_message("SimulationCollection", {"kind": "match", "dataset": "a"})
    assert not isinstance(result, SerdeError)


@pytest.mark.parametrize(
    ("target", "payload", "fragment"),
    [
        ("Nope", {"kind": "take", "n": 1}, "Unknown target type"),
        ("Dataset", {"n": 1}, "missing a string 'kind'"),
        ("Dataset", {"kind": 3}, "missing a string 'kind'"),
        ("Dataset", "[1, 2]", "must be an object"),
        ("Dataset", "{not json", ""),
    ],
)
def test_malformed_input_returns_error(target, payload, fragment):
    result = decode_message(target, payload)
    assert isinstance(result, SerdeError)
    assert fragment in result.message


def test_invalid_fields_report_field_errors():
    result = decode_message("Dataset", {"kind": "take", "n": -1})
    assert isinstance(result, SerdeError)
    assert result.category == SerdeErrorCategory.VALIDATION_ERROR
    assert result.field_errors[0].location == ("n",)


def test_matches_union_adapter():
    from opencosmo.serde import DatasetMessage

    payload = {"kind": "sort_by", "column": "mass", "invert": True}
    assert decode_message("Dataset", payload) == TypeAdapter(
        DatasetMessage
    ).validate_python(payload)
