from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

from astropy import cosmology
from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    import astropy.units as u
    import numpy as np


class FlrwCosmology(Protocol):
    """
    The part of an astropy FLRW cosmology that OpenCosmo uses.

    Astropy's ``dataclass_decorator`` is annotated as returning ``type[Cosmology]``,
    which hides ``H0`` and the other FLRW members from static type checkers.
    """

    @property
    def H0(self) -> u.Quantity: ...
    @property
    def h(self) -> float: ...
    def scale_factor(self, z: float | np.ndarray) -> float | np.ndarray: ...


def make_cosmology(parameters: CosmologyParameters) -> FlrwCosmology:
    # Every type from get_cosmology_type is an FLRW subclass; see FlrwCosmology.
    cosmology_type = cast(
        "Callable[..., FlrwCosmology]", get_cosmology_type(parameters)
    )
    expected_arguments = inspect.signature(cosmology_type).parameters.keys()
    input_paremeters = {}
    for argname in expected_arguments:
        try:
            input_paremeters[argname] = getattr(parameters, argname)
        except AttributeError:
            continue
    return cosmology_type(**input_paremeters, meta={"sigma_8": parameters.sigma_8})


def get_cosmology_type(parameters: CosmologyParameters) -> type[cosmology.Cosmology]:
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


class CosmologyParameters(BaseModel, frozen=True):
    """
    Responsible for validating cosmology parameters and handling differences in
    naming conventions between OpenCosmo and astropy.cosmology. Generally should
    not be used by the user directly
    """

    ACCESS_PATH: ClassVar[str] = "cosmology"
    ACCESS_TRANSFORMATION: ClassVar[Callable] = make_cosmology

    h: float = Field(ge=0.0, description="Reduced Hubble constant")

    @computed_field  # type: ignore[prop-decorator]
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
