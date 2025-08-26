from typing import Any, Optional, Type

from astropy.cosmology import Cosmology
from astropy.units.typing import UnitLike
from pydantic import BaseModel

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


def apply_units(
    model: Any, cosmology: Optional[Cosmology], convention: str = "scalefree"
):
    if not isinstance(model, BaseModel):
        return model
    model_units = __KNOWN_UNITFUL_MODELS__.get(type(model))

    parameters = model.model_dump()
    if model_units is not None:
        for name, unit in model_units[1].items():
            value = parameters[name] * unit
            parameters[name] = value

    return parameters
