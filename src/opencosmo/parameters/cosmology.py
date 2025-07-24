from typing import Callable, ClassVar

from pydantic import BaseModel, Field, computed_field

from opencosmo.cosmology import make_cosmology


class CosmologyParameters(BaseModel):
    """
    Responsible for validating cosmology parameters and handling differences in
    naming conventions between OpenCosmo and astropy.cosmology. Generally should
    not be used by the user directly
    """

    ACCESS_PATH: ClassVar[str] = "cosmology"
    ACCESS_TRANSFORMATION: ClassVar[Callable] = make_cosmology

    h: float = Field(ge=0.0, description="Reduced Hubble constant")

    @computed_field  # type: ignore
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
