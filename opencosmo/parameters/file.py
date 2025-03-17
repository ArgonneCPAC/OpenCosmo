from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator


class FileType(Enum):
    galaxy_properties = "galaxy_properties"
    halo_properties = "halo_properties"
    halo_profiles = "halo_profiles"
    halo_particles = "halo_particles"


class FileParameters(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    data_type: FileType
    is_lightcone: bool
    redshift: float
    step: int

    @field_validator("is_lightcone", mode="before")
    def validate_is_lightcone(cls, value):
        return bool(value)
