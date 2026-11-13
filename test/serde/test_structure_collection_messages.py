"""Generated StructureCollection messages, exercised without external data."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from unittest.mock import Mock

import astropy.units as u
import numpy as np
import opencosmo as oc
import pytest
from opencosmo.collection.structure import StructureCollection
from opencosmo.column.column import Column, ColumnMask, CompoundColumnMask
from opencosmo.dataset import Dataset
from opencosmo.serde import ParameterRole, message_signature
from opencosmo.serde.messages import structure_collection as messages
from pydantic import TypeAdapter, ValidationError

ADAPTER = TypeAdapter(messages.StructureCollectionMessage)
MODELS = {model.method: model for model in messages.STRUCTURE_COLLECTION_MESSAGE_TYPES}
SNAPSHOT = (
    Path(__file__).parent / "snapshots" / "structure_collection_messages.schema.json"
)
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
    "with_datasets",
    "with_redshift_range",
    "pixel_search",
}
ROLES: dict[inspect._ParameterKind, ParameterRole] = {
    inspect.Parameter.POSITIONAL_OR_KEYWORD: ParameterRole.POSITIONAL,
    inspect.Parameter.VAR_POSITIONAL: ParameterRole.VAR_ARGS,
    inspect.Parameter.KEYWORD_ONLY: ParameterRole.KEYWORD_ONLY,
    inspect.Parameter.VAR_KEYWORD: ParameterRole.VAR_KWARGS,
}
COLUMN = {"kind": "column", "name": "mass"}
DOUBLED = {
    "kind": "arithmetic",
    "operator": "multiply",
    "left": COLUMN,
    "right": {"kind": "number", "value": 2},
}
MEAN = {"kind": "reduction", "operator": "mean", "operand": COLUMN}
MASK = {
    "kind": "comparison",
    "operator": "greater_than",
    "left": COLUMN,
    "right": {"kind": "quantity", "value": 1, "unit": "Msun"},
}
COMPOUND = {"kind": "compound", "operator": "and", "left": MASK, "right": MASK}
BOX = {"kind": "box", "p1": [0, 0, 0], "p2": [10, 10, 10]}
PAYLOADS = [
    {"kind": "filter", "masks": [MASK, COMPOUND], "on_galaxies": True, "mode": "local"},
    {
        "kind": "select",
        "select_args": ["tag", ["mass"]],
        "mode": "local",
        "select_kwargs": {"m2": DOUBLED},
    },
    {
        "kind": "drop",
        "drop_args": ["tag", ["mass"]],
        "columns_to_drop": {"halo_properties": "tag"},
    },
    {"kind": "sort_by", "column": "tag", "invert": True},
    {"kind": "take", "n": 2, "at": "end", "mode": "global"},
    {"kind": "take_range", "start": 1, "end": 3, "mode": "global"},
    {"kind": "take_rows", "rows": [2, 0, 4]},
    {"kind": "bound", "region": BOX, "select_by": "galaxies"},
    {
        "kind": "with_new_columns",
        "dataset": "galaxies.galaxy_properties",
        "descriptions": {"m2": "doubled mass"},
        "allow_overwrite": True,
        "mode": "local",
        "new_columns": {"m2": DOUBLED},
    },
    {
        "kind": "with_units",
        "convention": "physical",
        "conversions": {"Msun": "kg"},
        "dataset_conversions": {
            "galaxies": {
                "galaxy_particles": {"mass": "g", "conversions": {"Mpc": "kpc"}}
            }
        },
    },
    {"kind": "with_datasets", "datasets": ["halo_properties", "galaxies"]},
    {"kind": "with_redshift_range", "z_low": 0.1, "z_high": 1.2},
    {"kind": "pixel_search", "pixels": [0, 11, 0], "nside": 1},
]


def test_all_thirteen_operations_are_exported() -> None:
    from opencosmo import serde

    assert set(MODELS) == METHODS
    assert len(messages.STRUCTURE_COLLECTION_MESSAGE_TYPES) == 13
    assert set(messages.__all__) <= set(serde.__all__)
    for name in messages.__all__:
        assert getattr(serde, name) is getattr(messages, name)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_message_mirrors_signature_and_defaults(method: str) -> None:
    model = MODELS[method]
    parameters = list(
        inspect.signature(getattr(StructureCollection, method)).parameters.values()
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


@pytest.mark.parametrize("payload", PAYLOADS, ids=lambda p: p["kind"])
def test_json_round_trip_and_apply_call_parity(payload: dict) -> None:
    message = ADAPTER.validate_json(json.dumps(payload))
    assert type(message) is MODELS[payload["kind"]]
    assert ADAPTER.validate_json(ADAPTER.dump_json(message)) == message
    assert ADAPTER.validate_python(message.model_dump(mode="json")) == message

    recorder = Mock(spec=StructureCollection)
    sentinel = object()
    getattr(recorder, message.method).return_value = sentinel
    assert message.apply(recorder) is sentinel
    assert len(recorder.mock_calls) == 1
    call = getattr(recorder, message.method).call_args
    match message.method:
        case "filter":
            assert len(call.args) == 2
            assert isinstance(call.args[0], ColumnMask)
            assert isinstance(call.args[1], CompoundColumnMask)
            assert call.kwargs == {"on_galaxies": True, "mode": "local"}
        case "select":
            assert call.args == ("tag", ("mass",))
            assert call.kwargs["mode"] == "local"
            assert isinstance(call.kwargs["m2"], Column)
            assert call.kwargs["m2"].uuid == (oc.col("mass") * 2).uuid
        case "drop":
            assert call.args == ("tag", ("mass",))
            assert call.kwargs == {"halo_properties": "tag"}
        case "take_rows" | "pixel_search":
            key = "rows" if message.method == "take_rows" else "pixels"
            np.testing.assert_array_equal(call.args[0], payload[key])
            assert call.args[0].dtype == np.int64
            assert call.args[0].ndim == 1
            assert call.args[1:] == (() if key == "rows" else (1,))
            assert call.kwargs == {}
        case "bound":
            expected = oc.make_box((0, 0, 0), (10, 10, 10))
            assert type(call.args[0]) is type(expected)
            assert call.args[0].into_model() == expected.into_model()
            assert call.args[1:] == ("galaxies",)
            assert call.kwargs == {}
        case "with_new_columns":
            assert call.args == (
                "galaxies.galaxy_properties",
                {"m2": "doubled mass"},
                True,
                "local",
            )
            assert set(call.kwargs) == {"m2"}
            assert call.kwargs["m2"].uuid == (oc.col("mass") * 2).uuid
        case "with_units":
            assert call.args == ("physical", {u.Msun: u.kg})
            assert call.kwargs == {
                "galaxies": {
                    "galaxy_particles": {"mass": u.g, "conversions": {u.Mpc: u.kpc}}
                }
            }
        case _:
            expected_args = {
                "sort_by": ("tag", True),
                "take": (2, "end", "global"),
                "take_range": (1, 3, "global"),
                "with_datasets": (("halo_properties", "galaxies"),),
                "with_redshift_range": (0.1, 1.2),
            }
            assert call.args == expected_args[message.method]
            assert call.kwargs == {}


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"kind": "take", "n": -1}, "greater than or equal to 0"),
        ({"kind": "take", "n": True}, "valid integer"),
        ({"kind": "take", "n": "3"}, "valid integer"),
        ({"kind": "take", "n": 1, "at": "middle"}, "'start', 'end' or 'random'"),
        ({"kind": "filter", "on_galaxies": 1}, "valid boolean"),
        ({"kind": "filter", "mode": "everywhere"}, "'local' or 'global'"),
        ({"kind": "filter", "masks": [DOUBLED]}, "does not match"),
        ({"kind": "take_range", "start": 4, "end": 1}, "end must be greater"),
        ({"kind": "take_range", "start": -1, "end": 1}, "greater than or equal to 0"),
        ({"kind": "take_rows", "rows": [-1]}, "greater than or equal to 0"),
        ({"kind": "take_rows", "rows": [True]}, "valid integer"),
        ({"kind": "take_rows", "rows": [[0]]}, "valid integer"),
        ({"kind": "sort_by"}, "Field required"),
        ({"kind": "sort_by", "column": None}, "valid string"),
        ({"kind": "sort_by", "column": "mass", "invert": "yes"}, "valid boolean"),
        ({"kind": "with_datasets", "datasets": [1]}, "valid string"),
        ({"kind": "with_redshift_range", "z_low": float("nan"), "z_high": 1}, "finite"),
        ({"kind": "with_redshift_range", "z_low": 0, "z_high": True}, "valid number"),
        ({"kind": "with_units", "convention": "metric"}, "'scalefree'"),
        (
            {"kind": "with_units", "conversions": {"bogus_unit": "kg"}},
            "Invalid Astropy",
        ),
        (
            {
                "kind": "with_units",
                "dataset_conversions": {
                    "galaxies": {"galaxy_particles": {"mass": "bogus_unit"}}
                },
            },
            "Invalid Astropy",
        ),
        (
            {
                "kind": "with_units",
                "dataset_conversions": {
                    "galaxies": {"galaxy_particles": {"mass": None}}
                },
            },
            "valid string",
        ),
        (
            {
                "kind": "with_units",
                "dataset_conversions": {
                    "galaxies": {"conversions": {"Msun": {"unit": "kg"}}}
                },
            },
            "valid string",
        ),
        ({"kind": "select", "select_kwargs": {"": COLUMN}}, "at least 1 character"),
        (
            {
                "kind": "select",
                "select_kwargs": {
                    "galaxies": {
                        "galaxy_properties": {"columns": ["mass"], "typo": "tag"}
                    }
                },
            },
            "Extra inputs",
        ),
        (
            {
                "kind": "select",
                "select_kwargs": {"m2": {**DOUBLED, "operator": "modulo"}},
            },
            "Input should be",
        ),
        (
            {
                "kind": "drop",
                "columns_to_drop": {"galaxies": {"galaxy_particles": [1]}},
            },
            "valid string",
        ),
        (
            {
                "kind": "with_new_columns",
                "dataset": "halo_properties",
                "new_columns": {"m2": MASK},
            },
            "does not match",
        ),
        (
            {
                "kind": "with_new_columns",
                "dataset": "halo_properties",
                "descriptions": {"other": "x"},
                "new_columns": {"m2": DOUBLED},
            },
            "unknown columns",
        ),
        ({"kind": "pixel_search", "pixels": [], "nside": 1}, "Pixels must be"),
        ({"kind": "pixel_search", "pixels": [0], "nside": 3}, "positive power of two"),
        ({"kind": "pixel_search", "pixels": [0], "nside": 0}, "positive power of two"),
        ({"kind": "pixel_search", "pixels": [12], "nside": 1}, "less than 12"),
        ({"kind": "pixel_search", "pixels": [-1]}, "greater than or equal to 0"),
        ({"kind": "pixel_search", "pixels": [True]}, "valid integer"),
        ({"kind": "pixel_search", "pixels": [0], "nside": True}, "valid integer"),
        ({"kind": "select", "extra": 1}, "Extra inputs"),
        ({"kind": "evaluate"}, "does not match"),
    ],
)
def test_invalid_payloads(payload: dict, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        ADAPTER.validate_python(payload)
    with pytest.raises(ValidationError, match=error):
        ADAPTER.validate_json(json.dumps(payload))


@pytest.mark.parametrize("kind", ["select", "with_new_columns"])
@pytest.mark.parametrize("expression", [MEAN, {"kind": "number", "value": 1}])
def test_apply_rejects_non_column_expressions(kind: str, expression: dict) -> None:
    payload = (
        {
            "kind": kind,
            "select_kwargs": {
                "halo_properties": {"derived_columns": {"m": expression}}
            },
        }
        if kind == "select"
        else {
            "kind": kind,
            "dataset": "halo_properties",
            "new_columns": {"m": expression},
        }
    )
    message = ADAPTER.validate_json(json.dumps(payload))
    recorder = Mock(spec=StructureCollection)
    with pytest.raises(TypeError, match="must produce a column"):
        message.apply(recorder)
    assert recorder.mock_calls == []


def test_recursive_select_and_drop(cached_structures: StructureCollection) -> None:
    payload = {
        "kind": "select",
        "select_kwargs": {
            "halo_properties": "mass",
            "galaxies": {
                "galaxy_properties": ["tag"],
                "galaxy_particles": {
                    "columns": ["mass"],
                    "derived_columns": {"m2": DOUBLED},
                },
            },
        },
    }
    result = ADAPTER.validate_json(json.dumps(payload)).apply(cached_structures)
    expected = cached_structures.select(
        halo_properties="mass",
        galaxies={
            "galaxy_properties": ("tag",),
            "galaxy_particles": {
                "columns": ("mass",),
                "derived_columns": {"m2": oc.col("mass") * 2},
            },
        },
    )
    assert (
        result["halo_properties"].columns
        == expected["halo_properties"].columns
        == ["mass"]
    )
    galaxies = result["galaxies"]
    assert isinstance(galaxies, StructureCollection)
    assert galaxies["galaxy_properties"].columns == ["tag"]
    leaf = galaxies["galaxy_particles"]
    expected_galaxies = expected["galaxies"]
    assert isinstance(expected_galaxies, StructureCollection)
    assert leaf.columns == expected_galaxies["galaxy_particles"].columns
    assert set(leaf.columns) == {"mass", "m2"}
    assert u.allclose(leaf.get_data()["m2"], np.array([2, 4, 8]) * u.Msun)

    drop = ADAPTER.validate_python(
        {"kind": "drop", "columns_to_drop": {"galaxies": {"galaxy_particles": "m2"}}}
    )
    dropped = drop.apply(result)
    dropped_galaxies = dropped["galaxies"]
    assert isinstance(dropped_galaxies, StructureCollection)
    assert dropped_galaxies["galaxy_particles"].columns == ["mass"]
    assert set(leaf.columns) == {"mass", "m2"}
    assert cached_structures["halo_properties"].columns == ["mass", "tag"]


def test_recursive_unit_normalization_and_application(
    cached_structures: StructureCollection,
) -> None:
    payload = {
        "kind": "with_units",
        "dataset_conversions": {
            "halo_properties": {"mass": "kg"},
            "galaxies": {
                "galaxy_properties": {"conversions": {"M_sun": "g"}},
                "galaxy_particles": {"mass": "kg", "conversions": {}},
            },
        },
    }
    message = ADAPTER.validate_json(json.dumps(payload))
    assert isinstance(message, messages.StructureCollectionWithUnitsMessage)
    galaxy_conversions = message.dataset_conversions["galaxies"]
    assert isinstance(galaxy_conversions, dict)
    property_conversions = galaxy_conversions["galaxy_properties"]
    assert isinstance(property_conversions, dict)
    assert property_conversions["conversions"] == {"solMass": "g"}
    assert ADAPTER.validate_json(message.model_dump_json()) == message
    result = message.apply(cached_structures)
    expected = cached_structures.with_units(
        halo_properties={"mass": u.kg},
        galaxies={
            "galaxy_properties": {"conversions": {u.Msun: u.g}},
            "galaxy_particles": {"mass": u.kg, "conversions": {}},
        },
    )
    for path, unit in [
        (("halo_properties",), u.kg),
        (("galaxies", "galaxy_properties"), u.g),
        (("galaxies", "galaxy_particles"), u.kg),
    ]:
        actual: Dataset | StructureCollection = result
        direct: Dataset | StructureCollection = expected
        original: Dataset | StructureCollection = cached_structures
        for key in path:
            assert isinstance(actual, StructureCollection)
            assert isinstance(direct, StructureCollection)
            assert isinstance(original, StructureCollection)
            actual, direct, original = actual[key], direct[key], original[key]
        assert isinstance(actual, Dataset)
        assert isinstance(direct, Dataset)
        assert isinstance(original, Dataset)
        actual_mass = actual.get_data(wrap_single=True)["mass"]
        original_mass = original.get_data(wrap_single=True)["mass"]
        assert isinstance(actual_mass, u.Quantity)
        assert isinstance(original_mass, u.Quantity)
        assert actual_mass.unit == unit
        assert u.allclose(actual_mass, direct.get_data(wrap_single=True)["mass"])
        assert original_mass.unit == u.Msun


def test_top_level_null_conversions_are_preserved_by_adapter() -> None:
    message = ADAPTER.validate_json(
        '{"kind": "with_units", "conversions": null, "dataset_conversions": '
        '{"galaxies": {"galaxy_particles": {"mass": "kg"}}}}'
    )
    assert ADAPTER.validate_json(message.model_dump_json()) == message
    recorder = Mock(spec=StructureCollection)
    message.apply(recorder)
    recorder.with_units.assert_called_once_with(
        None, None, galaxies={"galaxy_particles": {"mass": u.kg}}
    )


def test_nested_null_blanket_conversion_is_rejected() -> None:
    payload = (
        '{"kind": "with_units", "dataset_conversions": '
        '{"galaxies": {"galaxy_particles": {"mass": "kg", "conversions": null}}}}'
    )
    with pytest.raises(ValidationError, match="type=dict_type"):
        ADAPTER.validate_json(payload)
    with pytest.raises(ValidationError, match="type=dict_type"):
        ADAPTER.validate_python(json.loads(payload))


@pytest.mark.parametrize("selection", [{}, {"columns": []}, {"derived_columns": {}}])
def test_empty_selection_shapes_round_trip(selection: dict) -> None:
    message = ADAPTER.validate_python(
        {"kind": "select", "select_kwargs": {"galaxies": selection}}
    )
    assert ADAPTER.validate_json(message.model_dump_json()) == message
    recorder = Mock(spec=StructureCollection)
    message.apply(recorder)
    expected: dict[str, object] = (
        {} if not selection else {"columns": (), "derived_columns": {}}
    )
    assert recorder.select.call_args.kwargs == {"mode": "global", "galaxies": expected}


def test_empty_rows_produce_signed_one_dimensional_array() -> None:
    recorder = Mock(spec=StructureCollection)
    ADAPTER.validate_json('{"kind": "take_rows", "rows": []}').apply(recorder)
    rows = recorder.take_rows.call_args.args[0]
    assert rows.shape == (0,)
    assert rows.dtype == np.int64


@pytest.mark.parametrize("kind", ["select", "with_new_columns"])
@pytest.mark.parametrize("value", [np.array([1, 2]), np.array([1, 2]) * u.kg])
def test_live_column_data_cannot_be_sent_over_wire(kind: str, value: object) -> None:
    payload = (
        {"kind": kind, "select_kwargs": {"m2": value}}
        if kind == "select"
        else {"kind": kind, "dataset": "halo_properties", "new_columns": {"m2": value}}
    )
    with pytest.raises(ValidationError):
        ADAPTER.validate_python(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "take", "n": 0},
        {"kind": "take_range", "start": 0, "end": 0},
        {"kind": "pixel_search", "pixels": [0, 49151]},
        {"kind": "with_datasets", "datasets": "galaxies"},
    ],
)
def test_valid_boundaries_and_default_resolution(payload: dict) -> None:
    message = ADAPTER.validate_json(json.dumps(payload))
    assert ADAPTER.validate_json(message.model_dump_json()) == message


def test_nested_new_columns_and_dataset_subset(
    cached_structures: StructureCollection,
) -> None:
    message = ADAPTER.validate_json(
        json.dumps(
            {
                "kind": "with_new_columns",
                "dataset": "galaxies.galaxy_particles",
                "new_columns": {"m2": DOUBLED},
                "descriptions": {"m2": "doubled mass"},
            }
        )
    )
    result = message.apply(cached_structures)
    expected = cached_structures.with_new_columns(
        "galaxies.galaxy_particles",
        descriptions={"m2": "doubled mass"},
        m2=oc.col("mass") * 2,
    )
    galaxies = result["galaxies"]
    assert isinstance(galaxies, StructureCollection)
    leaf = galaxies["galaxy_particles"]
    expected_galaxies = expected["galaxies"]
    assert isinstance(expected_galaxies, StructureCollection)
    assert u.allclose(
        leaf.get_data()["m2"],
        expected_galaxies["galaxy_particles"].get_data()["m2"],
    )
    assert leaf.descriptions["m2"] == "doubled mass"
    subset = ADAPTER.validate_json(
        '{"kind": "with_datasets", "datasets": ["galaxies"]}'
    ).apply(result)
    assert subset.keys() == result.with_datasets(("galaxies",)).keys() == ["galaxies"]
    assert cached_structures.keys() == ["galaxies", "halo_properties"]
    original_galaxies = cached_structures["galaxies"]
    assert isinstance(original_galaxies, StructureCollection)
    assert original_galaxies["galaxy_particles"].columns == ["mass", "tag"]


def test_messages_are_frozen_and_have_independent_defaults() -> None:
    first = messages.StructureCollectionSelectMessage()
    second = messages.StructureCollectionSelectMessage()
    assert first.select_kwargs is not second.select_kwargs
    with pytest.raises(ValidationError, match="frozen"):
        first.mode = "local"


def test_structure_collection_message_schema_matches_snapshot() -> None:
    schema = json.dumps(ADAPTER.json_schema(), indent=2, sort_keys=True) + "\n"
    if os.environ.get("OPENCOSMO_UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(schema)
    assert SNAPSHOT.read_text() == schema, (
        "The StructureCollection message schema changed. If intended, rerun with "
        "OPENCOSMO_UPDATE_SNAPSHOTS=1 and review the snapshot diff."
    )
