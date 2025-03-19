from __future__ import annotations

from functools import cached_property
from typing import Optional

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opencosmo import parameters
from opencosmo.file import file_reader


@file_reader
def read_simulation_parameters(file: h5py.File) -> SimulationParameters:
    """
    Read the simulation parameters from an OpenCosmo file

    This function reads the simulation parameters from the header of an OpenCosmo file
    without reading the entire header or any data. It may be useful if you just want
    some basic information about the simulation.

    Parameters
    ----------

    """
    is_hydro = file["header/reformat_hacc/config"].attrs.get("is_hydro", None)
    if is_hydro is None:
        n_dm = file["header/simulation/parameters"].attrs.get("n_dm", 0)
        n_bar = file["header/simulation/parameters"].attrs.get("n_bar", 0)
        if n_dm > 0 and n_bar > 0:
            is_hydro = True
        elif n_dm > 0:
            is_hydro = False
        else:
            raise KeyError(
                "Could not determine if this simulation is hydro or gravity-only from "
                "the header. Are you sure it is an OpenCosmo file?"
            )

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
            file,
            "simulation/parameters",
            GravityOnlySimulationParameters,
            cosmology_parameters=cosmology_parameters,
        )


def empty_string_to_none(v):
    if v == "":
        return None
    return v


class SimulationParameters(BaseModel):
    box_size: float = Field(ge=0, description="Size of the simulation box (Mpc/h)")
    z_ini: float = Field(ge=0.01, description="Initial redshift")
    z_end: float = Field(ge=0.0, description="Final redshift")
    n_gravity: Optional[int] = Field(
        ge=2, description="Number of gravity-only particles (per dimension)"
    )
    n_steps: int = Field(ge=1, description="Number of time steps")
    pm_grid: int = Field(ge=2, description="Grid resolution (per dimension)")
    offset_gravity_ini: Optional[float] = Field(
        description="Lagrangian offset for gravity-only particles"
    )
    cosmology_parameters: parameters.CosmologyParameters = Field(
        description="Cosmology parameters",
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            return {k: empty_string_to_none(v) for k, v in data.items()}
        return data

    @cached_property
    def step_zs(self) -> list[float]:
        a_ini = 1 / (1 + self.z_ini)
        a_end = 1 / (1 + self.z_end)
        # Steps are evenly spaced in log(a)
        step_as = np.logspace(np.log10(a_ini), np.log10(a_end), self.n_steps)
        return np.round(1 / step_as - 1, 3).tolist()  # type: ignore


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
    n_dm: int = Field(
        ge=2, description="Number of dark matter particles (per dimension)"
    )
    offset_gas_ini: float = Field(
        description="Lagrangian offset for gas particles", alias="offset_bar_ini"
    )
    offset_dm_ini: float = Field(
        description="Lagrangian offset for dark matter particles"
    )
    subgrid_parameters: SubgridParameters = Field(
        description="Parameters for subgrid physics",
        exclude=True,
    )
