from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, field_serializer, field_validator, model_validator


def empty_string_to_none(value: str) -> Optional[str]:
    if type(value) is str and value == "":
        return None
    return value


class ReformatParamters(BaseModel):
    cosmotools_lc_path: Optional[Path] = None
    cosmotools_path: Path
    indat_path: Path
    is_hydro: bool
    lightcone_analysis_path_pattern: Optional[str] = None
    machine: str
    mass_threshold_sodbighaloparticles: Optional[float] = None
    mass_threshold_sodpropertybins: Optional[float] = None
    max_level: int = 0
    max_level_lc: Optional[list[tuple[int, int]]] = None
    npart_threshold_galaxyproperties: Optional[int] = None
    output_lc_path_pattern: Optional[str] = None
    rearrange_output_path_pattern: str
    rearrange_output_lc_path_pattern: Optional[str] = None
    simulation_date: date
    simulation_name: str
    snapshot_analysis_path_pattern: Optional[str] = None
    temporary_path: Optional[Path] = None

    @field_serializer(
        "cosmotools_lc_path",
        "cosmotools_path",
        "indat_path",
        "temporary_path",
        "lightcone_analysis_path_pattern",
        "output_lc_path_pattern",
        "rearrange_output_lc_path_pattern",
        "snapshot_analysis_path_pattern",
    )
    def handle_path(self, v):
        if v is None:
            return ""
        return str(v)

    @field_serializer("simulation_date")
    def handle_date(self, v):
        return v.isoformat()

    @field_validator("is_hydro", mode="before")
    @classmethod
    def numpy_bool_to_base(cls, value):
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            data = {k: empty_string_to_none(v) for k, v in data.items()}
        return data
