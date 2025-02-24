from typing import Optional

from pydantic import BaseModel, Field, model_validator

from opencosmo.file import file_reader

from .cosmology import CosmologyParameters


def empty_string_to_none(v):
    if v == "":
        return None
    return v


class SimulationParameters(BaseModel):
    box_size: float = Field(..., description="Size of the simulation box (Mpc/h)")
    z_ini: float = Field(description="Initial redshift")
    z_end: float = Field(description="Final redshift")
    n_dm: int = Field(description="Number of dark matter particles (per dimension)")
    n_gravity: Optional[int] = Field(
        ge=0, description="Number of gravity-only particles (per dimension)"
    )
    n_steps: int = Field(description="Number of time steps")
    pm_grid: int = Field(description="Grid resolution (per dimension)")
    offset_gravity_ini: Optional[float] = Field(
        description="Lagrangian offset for gravity-only particles"
    )
    offset_dm_ini: float = Field(
        description="Lagrangian offset for dark matter particles"
    )
    cosmology: CosmologyParameters = Field(description="Cosmology parameters")

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            return {k: empty_string_to_none(v) for k, v in data.items()}
        return data


GravityOnlySimulationParameters = SimulationParameters


class SubgridParameters(BaseModel):
    agn_kinetic_eps: float = Field(description="AGN feedback efficiency")
    agn_kinetic_jet_vel: float = Field(description="AGN feedback velocity")
    agn_nperh: float = Field(description="AGN sphere of influence")
    agn_seed_mass: float = Field(description="AGN seed mass")
    wind_egy_w: float = Field(description="Wind mass loading factor")
    wind_kappa_w: float = Field(description="Wind belovity")


class HydroSimulationParameters(SimulationParameters):
    n_gas: int = Field(
        description="Number of gas particles (per dimension)", alias="n_bar"
    )
    offset_gas_ini: float = Field(
        description="Lagrangian offset for gas particles", alias="offset_bar_ini"
    )
    subgrid_params: SubgridParameters = Field(
        description="Parameters for subgrid physics"
    )
