# ruff: noqa: TC001 TC003
from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from opencosmo.spatial import models as sm
from opencosmo.units import UnitConvention


def empty_string_to_none(value: str) -> Optional[str]:
    if type(value) is str and value == "":
        return None
    return value


class FileType(Enum):
    galaxy_properties = "galaxy_properties"
    galaxy_particles = "galaxy_particles"
    halo_properties = "halo_properties"
    halo_profiles = "halo_profiles"
    halo_particles = "halo_particles"
    diffsky_fits = "diffsky_fits"
    healpix_map = "healpix_map"


class FileParameters(BaseModel):
    model_config = ConfigDict(use_enum_values=True, frozen=True)
    origin: str = "HACC"
    data_type: FileType
    is_lightcone: bool
    redshift: Optional[float] = None
    step: Optional[int] = None
    region: Optional[sm.RegionModel] = None
    unit_convention: UnitConvention = UnitConvention.SCALEFREE

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            data = {k: empty_string_to_none(v) for k, v in data.items()}
        return data

    @model_validator(mode="before")
    def parse_region(cls, data):
        region_keys = list(filter(lambda k: k.startswith("region_"), data.keys()))
        region_dict = {k[7:]: data.pop(k) for k in region_keys}
        if region_dict:
            data.update({"region": region_dict})

        return data

    @field_validator("is_lightcone", mode="before")
    def validate_is_lightcone(cls, value):
        return bool(value)

    @field_validator("unit_convention", mode="before")
    def validate_convention(cls, value):
        if isinstance(value, str):
            return UnitConvention(value)
        return value

    @field_serializer("unit_convention")
    def serialize_convention(self, value):
        if isinstance(value, UnitConvention):
            return value.value
        return value

    @model_serializer(mode="wrap")
    def serialize_model(self, handle):
        dump = handle(self)
        if dump["region"] is not None:
            region = dump.pop("region")
            region = {f"region_{k}": v for k, v in region.items()}
            dump.update(region)
        return dump
