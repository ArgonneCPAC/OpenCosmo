from __future__ import annotations

from typing import Optional, Type

import h5py
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opencosmo import parameters
from opencosmo.file import file_reader


@file_reader
def read_simulation_parameters(
    file: h5py.File,
    is_hydro: bool,
) -> Type[SimulationParameters]:
    try:
        cosmology_parameters = parameters.read_header_attributes(
            file, "simulation/cosmology", parameters.CosmologyParameters
        )
    except KeyError as e:
        raise KeyError(
            "This file does not appear to have cosmology information. "
            "Are you sure it is an OpenCosmo file?\n"
            f"Error: {e}"
        )

    if is_hydro:
        subrid_params = parameters.read_header_attributes(
            file,
            "simulation/parameters",
            SubgridParameters,
            cosmology_parameters=cosmology_parameters,
        )
        return parameters.read_header_attributes(
            file,
            "simulation/parameters",
            HydroSimulationParameters,
            subgrid_parameters=subrid_params,
            cosmology_parameters=cosmology_parameters,
        )
    else:
        return parameters.read_header_attributes(
            file, "simulation/paramters", GravityOnlySimulationParameters
        )


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
    cosmology_parameters: parameters.CosmologyParameters = Field(
        description="Cosmology parameters"
    )

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            return {k: empty_string_to_none(v) for k, v in data.items()}
        return data


GravityOnlySimulationParameters = SimulationParameters


def subgrid_alias_generator(name: str) -> str:
    return f"subgrid_{name}"


class SubgridParameters(BaseModel):
    model_config = ConfigDict(alias_generator=subgrid_alias_generator)
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
    subgrid_parameters: SubgridParameters = Field(
        description="Parameters for subgrid physics"
    )
