"""Strict recursive wire parameters for structure collections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BeforeValidator, ConfigDict, Field, TypeAdapter, model_validator

from opencosmo.dataset import Dataset

from .checks import explicit_parameters, reserved_keywords
from .common import Keyword, UnitString
from .expression import Expression, ExpressionModel

if TYPE_CHECKING:
    from pydantic import GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema

_SELECT_PARAMETERS = explicit_parameters(Dataset.select)
_UNIT_PARAMETERS = tuple(
    name for name in explicit_parameters(Dataset.with_units) if name != "conversions"
)


@dataclass(frozen=True)
class _ReservedKeys:
    names: tuple[str, ...]

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        result = handler.resolve_ref_schema(handler(schema))
        result["propertyNames"] = {
            "allOf": [
                result.get("propertyNames", {}),
                {"not": {"enum": list(self.names)}},
            ]
        }
        return result


@dataclass(frozen=True)
class _UnitSettingsSchema:
    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        result = handler.resolve_ref_schema(handler(schema))
        result["properties"] = {"conversions": handler(_UNIT_CONVERSIONS.core_schema)}
        return result


type ColumnNames = str | tuple[str, ...]


class DerivedSelection(ExpressionModel):
    """An explicit dataset selection with optional derived expressions."""

    columns: tuple[str, ...] = ()
    derived_columns: Annotated[
        dict[Keyword, Expression], _ReservedKeys(_SELECT_PARAMETERS)
    ] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "anyOf": [{"required": ["columns"]}, {"required": ["derived_columns"]}]
        }
    )

    @model_validator(mode="after")
    def check_derived_keywords(self) -> Self:
        """Reject names consumed by the live dataset selection signature."""
        reserved_keywords(self.derived_columns, _SELECT_PARAMETERS)
        return self

    @model_validator(mode="before")
    @classmethod
    def require_selection_fields(cls, value: object) -> object:
        """Keep empty nested mappings distinct from explicit dataset selections."""
        if isinstance(value, dict) and not {"columns", "derived_columns"}.intersection(
            value
        ):
            raise ValueError("Explicit selections require columns or derived_columns")
        return value


def __nested_selection(value: object) -> object:
    if isinstance(value, dict) and {"columns", "derived_columns", "kind"}.intersection(
        value
    ):
        raise ValueError(
            "Selections with reserved keys must match an explicit selection or expression"
        )
    return value


type StructureSelection = (
    ColumnNames
    | Expression
    | DerivedSelection
    | Annotated[
        dict[Keyword, StructureSelection],
        BeforeValidator(__nested_selection),
        _ReservedKeys(("columns", "derived_columns", "kind")),
    ]
)
type StructureDrop = ColumnNames | dict[Keyword, StructureDrop]
type UnitConversions = dict[UnitString, UnitString]

_UNIT_CONVERSIONS: TypeAdapter[UnitConversions] = TypeAdapter(UnitConversions)


def __unit_settings(value: object) -> object:
    if not isinstance(value, dict):
        return value
    reserved_keywords(value, _UNIT_PARAMETERS)
    result: dict[str, object] = {}
    for name, setting in value.items():
        if name == "conversions":
            setting = _UNIT_CONVERSIONS.validate_python(setting)
        result[name] = setting
    return result


type StructureUnitSettings = Annotated[
    dict[Keyword, UnitString | StructureUnitSettings],
    BeforeValidator(__unit_settings),
    _UnitSettingsSchema(),
    _ReservedKeys(_UNIT_PARAMETERS),
]

__all__ = [
    "ColumnNames",
    "DerivedSelection",
    "StructureDrop",
    "StructureSelection",
    "StructureUnitSettings",
    "UnitConversions",
]
