"""Recursive wire schemas preserve parser shape and reserved-key constraints."""

import pytest
from opencosmo.serde.structure import (
    DerivedSelection,
    StructureSelection,
    StructureUnitSettings,
)
from pydantic import TypeAdapter, ValidationError


def test_selection_schema_constraints_and_recursive_refs() -> None:
    schema = TypeAdapter(StructureSelection).json_schema()
    definitions = schema["$defs"]
    nested = definitions["StructureSelection"]["anyOf"][-1]
    assert nested["additionalProperties"] == {"$ref": "#/$defs/StructureSelection"}
    assert nested["propertyNames"]["allOf"] == [
        {"$ref": "#/$defs/Keyword"},
        {"not": {"enum": ["columns", "derived_columns", "kind"]}},
    ]
    explicit = definitions["DerivedSelection"]
    assert explicit["additionalProperties"] is False
    assert explicit["anyOf"] == [
        {"required": ["columns"]},
        {"required": ["derived_columns"]},
    ]
    assert explicit["properties"]["derived_columns"]["propertyNames"]["allOf"] == [
        {"$ref": "#/$defs/Keyword"},
        {"not": {"enum": ["mode"]}},
    ]
    with pytest.raises(ValidationError, match="require columns or derived_columns"):
        DerivedSelection.model_validate({})


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_unit_schema_constraints_and_recursive_refs(mode: str) -> None:
    schema = TypeAdapter(StructureUnitSettings).json_schema(mode=mode)
    settings = schema["$defs"]["StructureUnitSettings"]
    assert settings["propertyNames"]["allOf"] == [
        {"$ref": "#/$defs/Keyword"},
        {"not": {"enum": ["convention"]}},
    ]
    assert settings["additionalProperties"] == {
        "anyOf": [
            {"$ref": "#/$defs/UnitString"},
            {"$ref": "#/$defs/StructureUnitSettings"},
        ]
    }
    conversions = settings["properties"]["conversions"]
    assert conversions == {"$ref": "#/$defs/UnitConversions"}
    assert schema["$defs"]["UnitConversions"] == {
        "type": "object",
        "propertyNames": {"$ref": "#/$defs/UnitString"},
        "additionalProperties": {"$ref": "#/$defs/UnitString"},
    }


@pytest.mark.parametrize(
    ("wire_type", "valid", "invalid"),
    [
        (
            StructureSelection,
            [
                {},
                {"columns": []},
                {"derived_columns": {}},
                {"galaxies": {"leaf": "mass"}},
            ],
            [
                {"columns": "mass"},
                {"derived_columns": "mass"},
                {"kind": "mass"},
                {"galaxies": {"columns": "mass"}},
                {"derived_columns": {"mode": {"kind": "column", "name": "mass"}}},
            ],
        ),
        (
            StructureUnitSettings,
            [
                {},
                {"mass": "kg"},
                {"conversions": {}},
                {"leaf": {"conversions": {"Msun": "kg"}}},
            ],
            [
                {"mass": None},
                {"conversions": None},
                {"conversions": "kg"},
                {"conversions": {"Msun": {"mass": "kg"}}},
                {"leaf": {"conversions": None}},
                {"leaf": {"convention": "kg"}},
            ],
        ),
        (DerivedSelection, [{"columns": []}, {"derived_columns": {}}], [{}]),
    ],
)
def test_jsonschema_validation_agrees_with_parser(
    wire_type: object, valid: list[dict], invalid: list[dict]
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    adapter = TypeAdapter(wire_type)
    schema = adapter.json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for value in valid:
        validator.validate(value)
        adapter.validate_python(value)
    for value in invalid:
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(value)
        with pytest.raises(ValidationError):
            adapter.validate_python(value)
