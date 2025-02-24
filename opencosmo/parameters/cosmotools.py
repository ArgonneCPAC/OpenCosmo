from typing import Optional

from pydantic import BaseModel


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
