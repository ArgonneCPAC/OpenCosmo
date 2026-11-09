"""Parsing decoded JSON into messages and applying them to OpenCosmo objects."""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np
import opencosmo as oc
import pytest
from opencosmo.serde import (
    BoxRegionMessage,
    DatasetTakeMessage,
    LightconePixelSearchMessage,
    LightconeTakeMessage,
    MessageError,
    StructureCollectionTakeMessage,
    apply_message,
    execute_message,
    parse_message,
)

if TYPE_CHECKING:
    from opencosmo.collection.structure import StructureCollection
    from pydantic import JsonValue

_TRANSFORMS = ("select", "with_units", "with_new_columns")


def __live_target(
    request: pytest.FixtureRequest, target_type: type
) -> oc.Dataset | oc.StructureCollection:
    return request.getfixturevalue(
        "cached_dataset" if target_type is oc.Dataset else "cached_structures"
    )


def __spy_on_transforms(target_type: type, stack: ExitStack) -> list[Mock]:
    return [
        stack.enter_context(patch.object(target_type, name)) for name in _TRANSFORMS
    ]


TAKE: dict[str, JsonValue] = {"kind": "take", "n": 5, "at": "start"}


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


@pytest.mark.parametrize("target_name", ["cached_dataset", "cached_structures"])
def test_data_free_parse_selects_target_union(
    request: pytest.FixtureRequest, target_name: str
) -> None:
    target = request.getfixturevalue(target_name)
    model = (
        DatasetTakeMessage
        if target_name == "cached_dataset"
        else StructureCollectionTakeMessage
    )
    assert parse_message(type(target), json.loads(json.dumps(TAKE))) == model(
        n=5, at="start"
    )


def test_execute_structure_columns_matches_direct_api(
    cached_structures: StructureCollection,
) -> None:
    payload: dict[str, JsonValue] = {
        "kind": "select",
        "select_kwargs": {
            "halo_properties": "mass",
            "galaxies": {"galaxy_particles": ["tag"]},
        },
    }
    result = execute_message(cached_structures, payload)
    expected = cached_structures.select(
        halo_properties="mass", galaxies={"galaxy_particles": ("tag",)}
    )
    assert isinstance(result, oc.StructureCollection)
    assert (
        result["halo_properties"].columns
        == expected["halo_properties"].columns
        == ["mass"]
    )
    galaxies = result["galaxies"]
    assert isinstance(galaxies, oc.StructureCollection)
    assert galaxies["galaxy_particles"].columns == ["tag"]
    np.testing.assert_array_equal(
        result["halo_properties"].get_data(), expected["halo_properties"].get_data()
    )
    assert cached_structures["halo_properties"].columns == ["mass", "tag"]


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


@pytest.mark.parametrize(
    ("target_name", "message"),
    [
        ("cached_dataset", StructureCollectionTakeMessage(n=1)),
        ("cached_structures", DatasetTakeMessage(n=1)),
    ],
)
def test_apply_rejects_cross_target_operation_messages(
    request: pytest.FixtureRequest,
    target_name: str,
    message: DatasetTakeMessage | StructureCollectionTakeMessage,
) -> None:
    target = request.getfixturevalue(target_name)
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


@pytest.mark.parametrize("target_type", [oc.HealpixMap, oc.SimulationCollection])
def test_collection_targets_without_messages_are_rejected(target_type):
    target = Mock(spec=target_type)
    for result in (
        parse_message(target_type, TAKE),
        apply_message(target, LightconeTakeMessage(n=1)),
        apply_message(target, StructureCollectionTakeMessage(n=1)),
        execute_message(target, TAKE),
    ):
        assert isinstance(result, MessageError)
        assert result.kind == "unsupported_target"
    assert target.mock_calls == []


@pytest.mark.parametrize(
    ("target_name", "payload"),
    [
        (
            "cached_dataset",
            {"kind": "select", "select_kwargs": {"halo_properties": "mass"}},
        ),
        ("cached_structures", {"kind": "select", "columns": ["mass"]}),
        ("cached_dataset", {"kind": "with_datasets", "datasets": "halo_properties"}),
        ("cached_structures", {"kind": "pixel_search", "pixels": [12], "nside": 1}),
    ],
)
def test_parse_and_execute_reject_target_specific_invalid_payloads(
    request: pytest.FixtureRequest, target_name: str, payload: dict
) -> None:
    target = request.getfixturevalue(target_name)
    parsed = parse_message(type(target), payload)
    assert isinstance(parsed, MessageError)
    assert parsed.kind == "invalid_message"
    assert parsed.target == type(target).__name__
    assert execute_message(target, payload) == parsed


@pytest.mark.parametrize("target_type", [oc.Dataset, oc.StructureCollection])
def test_execute_spy_uses_target_specific_message(
    request: pytest.FixtureRequest, target_type: type
) -> None:
    target = __live_target(request, target_type)
    sentinel = object()
    with patch.object(target_type, "take", autospec=True) as take:
        take.return_value = sentinel
        assert execute_message(target, TAKE) is sentinel
    take.assert_called_once_with(target, 5, "start", "local")


def test_structure_operation_errors_are_captured(
    cached_structures: StructureCollection,
) -> None:
    result = execute_message(
        cached_structures, {"kind": "with_datasets", "datasets": "missing"}
    )

    assert isinstance(result, MessageError)
    assert result.kind == "operation_failed"
    assert result.target == "StructureCollection"
    assert result.error_type == "ValueError"
    assert "Unknown datasets" in result.detail


@pytest.mark.parametrize("target_type", [oc.Dataset, oc.StructureCollection])
@pytest.mark.parametrize("reserved", ["mode", "descriptions", "allow_overwrite"])
def test_new_column_keyword_collisions_fail_before_execution(
    request: pytest.FixtureRequest, target_type: type, reserved: str
) -> None:
    target = __live_target(request, target_type)
    payload: dict[str, JsonValue] = {
        "kind": "with_new_columns",
        "new_columns": {reserved: {"kind": "column", "name": "mass"}},
    }
    if target_type is oc.StructureCollection:
        payload["dataset"] = "halo_properties"
    parsed = parse_message(target_type, payload)
    assert isinstance(parsed, MessageError)
    assert parsed.kind == "invalid_message"
    assert "Reserved keyword names" in parsed.detail
    with ExitStack() as stack:
        spies = __spy_on_transforms(target_type, stack)
        assert execute_message(target, payload) == parsed
    assert [spy.mock_calls for spy in spies] == [[]] * len(spies)


@pytest.mark.parametrize(
    ("target_type", "payload"),
    [
        (
            oc.Dataset,
            {
                "kind": "select",
                "derived_columns": {"mode": {"kind": "column", "name": "mass"}},
            },
        ),
        (
            oc.StructureCollection,
            {"kind": "select", "select_kwargs": {"mode": "mass"}},
        ),
        *[
            (target_type, {"kind": "with_units", field: {reserved: "kg"}})
            for target_type, field in [
                (oc.Dataset, "columns"),
                (oc.StructureCollection, "dataset_conversions"),
            ]
            for reserved in ["convention", "conversions"]
        ],
        (
            oc.StructureCollection,
            {
                "kind": "select",
                "select_kwargs": {
                    "galaxies": {
                        "galaxy_properties": {
                            "derived_columns": {
                                "mode": {"kind": "column", "name": "mass"}
                            }
                        }
                    }
                },
            },
        ),
        (
            oc.StructureCollection,
            {
                "kind": "with_units",
                "dataset_conversions": {
                    "galaxies": {"galaxy_properties": {"convention": "kg"}}
                },
            },
        ),
        (
            oc.StructureCollection,
            {
                "kind": "with_units",
                "dataset_conversions": {
                    "galaxies": {"galaxy_properties": {"conversions": None}}
                },
            },
        ),
    ],
)
def test_reserved_or_null_recursive_keywords_are_invalid_messages(
    request: pytest.FixtureRequest, target_type: type, payload: dict
) -> None:
    target = __live_target(request, target_type)
    with ExitStack() as stack:
        spies = __spy_on_transforms(target_type, stack)
        for decoded in (payload, json.loads(json.dumps(payload))):
            parsed = parse_message(target_type, decoded)
            assert isinstance(parsed, MessageError)
            assert parsed.kind == "invalid_message"
            assert execute_message(target, decoded) == parsed
    assert [spy.mock_calls for spy in spies] == [[]] * len(spies)


@pytest.mark.parametrize("target_name", ["cached_dataset", "cached_structures"])
def test_execute_captures_operation_errors(
    request: pytest.FixtureRequest, target_name: str
) -> None:
    target = request.getfixturevalue(target_name)
    payload: dict[str, JsonValue] = {
        "kind": "filter",
        "masks": [
            {
                "kind": "comparison",
                "operator": "greater_than",
                "left": {"kind": "column", "name": "no_such_column"},
                "right": {"kind": "number", "value": 1.0},
            }
        ],
    }
    result = execute_message(target, payload)

    assert isinstance(result, MessageError)
    assert result.kind == "operation_failed"
    assert result.target == type(target).__name__
    assert result.error_type is not None
    assert "no_such_column" in result.detail
    # The error must serialize for the trip back to the client
    assert json.loads(result.model_dump_json())["kind"] == "operation_failed"


def test_apply_captures_operation_errors(cached_dataset) -> None:
    payload: dict[str, JsonValue] = {
        "kind": "filter",
        "masks": [
            {
                "kind": "comparison",
                "operator": "greater_than",
                "left": {"kind": "column", "name": "no_such_column"},
                "right": {"kind": "number", "value": 1.0},
            }
        ],
    }
    message = parse_message(type(cached_dataset), json.loads(json.dumps(payload)))
    result = apply_message(cached_dataset, message)
    assert isinstance(result, MessageError)
    assert result.kind == "operation_failed"
    assert result.error_type is not None
