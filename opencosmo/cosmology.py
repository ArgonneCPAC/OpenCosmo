import inspect
from functools import singledispatch
from pathlib import Path
from typing import Type

import h5py
from astropy import cosmology
from pydantic import BaseModel, Field, computed_field

from opencosmo.file import oc_reader

"""
Reads cosmology from the header of the file and returns the
astropy.cosmology object.
"""


class CosmologyParameters(BaseModel):
    h: float = Field(ge=0.0, description="Reduced Hubble constant")

    @computed_field
    @property
    def H0(self) -> float:
        """
        Hubble constant in km/s/Mpc
        """
        return self.h * 100

    Om0: float = Field(ge=0.0, description="Total matter density", alias="omega_m")
    Ob0: float = Field(ge=0.0, description="Baryon density", alias="omega_b")
    Ode0: float = Field(ge=0.0, description="Dark energy density", alias="omega_l")
    Neff: float = Field(
        gt=0.0, description="Effective number of neutrinos", alias="n_eff_massless"
    )
    n_eff_massive: float = Field(
        0, ge=0.0, description="Effective number of massive neutrinos"
    )
    sigma_8: float = Field(ge=0.0, description="RMS mass fluctuation at 8 Mpc/h")
    w0: float = Field(description="Dark energy equation of state", alias="w_0")
    wa: float = Field(
        description="Dark energy equation of state evolution", alias="w_a"
    )


def get_cosmology_type(parameters: CosmologyParameters) -> Type[cosmology.Cosmology]:
    is_flat = (parameters.Om0 + parameters.Ode0) == 1.0
    if parameters.w0 == -1 and parameters.wa == 0:
        if is_flat:
            return cosmology.FlatLambdaCDM
        else:
            return cosmology.LambdaCDM
    if parameters.w0 != -1 and parameters.wa == 0:
        if is_flat:
            return cosmology.FlatwCDM
        else:
            return cosmology.wCDM
    if parameters.wa != 0:
        if is_flat:
            return cosmology.Flatw0waCDM
        else:
            return cosmology.w0waCDM
    raise ValueError("Could not determine cosmology type.")


@oc_reader
def read_cosmology(file: h5py.File) -> cosmology.Cosmology:
    try:
        cosmology_data = file["header"]["simulation"]["cosmology"].attrs
    except (KeyError, AttributeError):
        raise KeyError(
            "This file does not appear to have cosmology information. "
            "Are you sure it is an OpenCosmo file?"
        )
    parameters = CosmologyParameters(**cosmology_data)
    cosmology_type = get_cosmology_type(parameters)
    expected_arguments = inspect.signature(cosmology_type).parameters.keys()
    input_paremeters = {}
    for argname in expected_arguments:
        try:
            input_paremeters[argname] = getattr(parameters, argname)
        except AttributeError:
            continue
    return cosmology_type(**input_paremeters)
