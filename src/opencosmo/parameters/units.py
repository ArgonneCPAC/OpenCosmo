from typing import Optional, Type

from astropy.cosmology import Cosmology
from astropy.units.typing import UnitLike
from pydantic import BaseModel

import opencosmo.transformations as t
from opencosmo.transformations.protocols import ColumnTransformation, TransformationType
from opencosmo.transformations.units import (
    apply_unit,
    get_unit_transition_transformations,
)

ModelUnitAnnotation = tuple[str, dict[str, UnitLike]]

__KNOWN_UNITFUL_MODELS__: dict[Type[BaseModel], ModelUnitAnnotation] = {}


# Constraint: Unit covention for all fields in a given model must be the same


def register_units(
    model: Type[BaseModel],
    field_name: str,
    unit: UnitLike,
    convention: str = "scalefree",
):
    model_spec = __KNOWN_UNITFUL_MODELS__.get(model)
    registered_fields: dict[str, UnitLike]
    if model_spec is not None and model_spec[0] != convention:
        raise ValueError(
            "All unitful fields in a parameter model must use the same unit convention"
        )
    elif model_spec is None:
        registered_fields = {}
    else:
        registered_fields = model_spec[1]
    if field_name in registered_fields:
        raise ValueError(f"Field {field_name} was already registered with units!")

    registered_fields[field_name] = unit
    __KNOWN_UNITFUL_MODELS__[model] = (convention, registered_fields)


def __get_unit_transformations(
    model: BaseModel, cosmology, convention: str = "scalefree"
) -> t.TransformationDict:
    transformations: t.TransformationDict = {}
    if (us := __KNOWN_UNITFUL_MODELS__.get(type(model))) is None:
        return {}

    base_convention, known_units = us

    column_transformations: list[ColumnTransformation] = []
    for name, unit in known_units.items():
        column_transformations.append(apply_unit(name, unit))

    transformations[TransformationType.COLUMN] = column_transformations

    transformations = get_unit_transition_transformations(
        base_convention, convention, transformations, cosmology
    )
    return transformations


def apply_units(
    model: BaseModel, cosmology: Optional[Cosmology], convention: str = "scalefree"
):
    transformations = __get_unit_transformations(model, cosmology, convention)
    parameters = model.model_dump()
    column_transformations = transformations.get(TransformationType.COLUMN, [])
    all_column_transformations = transformations.get(TransformationType.ALL_COLUMNS, [])
    for trans in column_transformations:
        assert hasattr(trans, "column_name")
        value = trans(parameters[trans.column_name])

        for trans_ in all_column_transformations:
            value = trans_(value)
        parameters[trans.column_name] = value
    return parameters
