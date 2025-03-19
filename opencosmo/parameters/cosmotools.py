from typing import Optional

from pydantic import BaseModel, model_validator


def empty_string_to_none(value: str) -> Optional[str]:
    if type(value) is str and value == "":
        return None
    return value


class CosmoToolsParameters(BaseModel):
    cosmotools_steps: list[int]
    fof_linking_length: float
    fof_pmin: int
    sod_pmin: int
    sod_delta_crit: float
    sod_concentration_pmin: int
    sodbighaloparticles_pmin: int
    profiles_nbins: int
    galaxy_dbscan_neighbors: Optional[int]
    galaxy_aperture_radius: Optional[int]
    galaxy_pmin: Optional[int]

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            data = {k: empty_string_to_none(v) for k, v in data.items()}
        return data
