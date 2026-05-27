from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

from astropy.units.typing import UnitLike

from opencosmo.units import UnitConvention
from opencosmo.units.get import get_unit_applicators_dict

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology
    from pydantic import BaseModel

ModelUnitAnnotation = tuple[UnitConvention, dict[str, UnitLike], bool]

__KNOWN_UNITFUL_MODELS__: dict[Type[BaseModel], ModelUnitAnnotation] = {}


# Constraint: Unit covention for all fields in a given model must be the same


def register_units(
    model: Type[BaseModel],
    field_name: str,
    unit: UnitLike,
    convention: UnitConvention = UnitConvention.SCALEFREE,
    is_comoving: bool = True,
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
    __KNOWN_UNITFUL_MODELS__[model] = (convention, registered_fields, is_comoving)


def __get_unit_transformations(
    model: BaseModel, cosmology, convention: UnitConvention = UnitConvention.SCALEFREE
) -> dict:
    if (us := __KNOWN_UNITFUL_MODELS__.get(type(model))) is None:
        return {}
    base_convention, known_units, is_comoving = us
    applicators = get_unit_applicators_dict(
        known_units, base_convention, cosmology, is_comoving
    )
    return applicators


def apply_units(
    model: BaseModel,
    cosmology: Cosmology,
    convention: UnitConvention = UnitConvention.SCALEFREE,
    unit_kwargs: dict[str, Any] = {},
):
    applicators = __get_unit_transformations(model, cosmology, convention)
    parameters = model.model_dump()
    for name, applicator in applicators.items():
        parameters[name] = applicator.apply(
            parameters[name], convention, unit_kwargs=unit_kwargs
        )
    return parameters
