from typing import Any, Callable, Optional

import astropy.cosmology.units as cu
import astropy.units as u
import h5py
from astropy.constants import m_p
from numpy.typing import ArrayLike

from opencosmo.header import OpenCosmoHeader
from opencosmo.units.convention import UnitConvention
from opencosmo.units.converters import get_unit_transitions

KNOWN_UNITS = {
    "comoving Mpc/h": u.Mpc / cu.littleh,
    "comoving (Mpc/h)^2": (u.Mpc / cu.littleh) ** 2,
    "comoving km/s": u.km / u.s,
    "comoving (km/s)^2": (u.km / u.s) ** 2,
    "Msun/h": u.Msun / cu.littleh,
    "Msun/yr": u.Msun / u.yr,
    "K": u.K,
    "comoving (Msun/h * (km/s) * Mpc/h)": (u.Msun / cu.littleh)
    * (u.km / u.s)
    * (u.Mpc / cu.littleh),
    "log10(erg/s)": u.DexUnit("erg/s"),
    "h^2 keV / (comoving cm)^3": (cu.littleh**2) * u.keV / (u.cm**3),
    "keV * cm^2": u.keV * u.cm**2,
    "cm^-3": u.cm**-3,
    "Gyr": u.Gyr,
    "Msun/h / (comoving Mpc/h)^3": (u.Msun / cu.littleh) / (u.Mpc / cu.littleh) ** 3,
    "Msun/h * km/s": (u.Msun / cu.littleh) * (u.km / u.s),
    "H0^-1": (u.s * (1 * u.Mpc).to(u.km).value).to(u.year) / (100 * cu.littleh),
    "m_hydrogen": m_p,
    "Msun * (km/s)^2": (u.Msun) * (u.km / u.s) ** 2,
}


class UnitApplicator:
    def __init__(
        self,
        base_unit: Optional[u.Unit],
        base_convention: UnitConvention,
        converters: dict[str, Callable],
    ):
        self.__base_unit = base_unit
        self.__base_convention = UnitConvention
        self.__converters = converters

    def apply(
        self,
        value: ArrayLike,
        convention: UnitConvention,
        convert_to: Optional[u.Unit] = None,
        unit_kwargs: dict[str, Any] = {},
    ) -> u.Quantity:
        if self.__base_unit is None or convention == UnitConvention.UNITLESS:
            return value
        if hasattr(value, "unit"):
            raise ValueError(
                "Units can only be applied to unitless scalars and numpy arrays"
            )
        new_value = value * self.__base_unit
        if convention != self.__base_convention:
            new_value = self.__convert(new_value, convention, unit_kwargs)

        if convert_to is not None:
            new_value = new_value.to(convert_to)

        return new_value

    def __convert(
        self, value: u.Quantity, to_: UnitConvention, unit_kwargs: dict[str, Any]
    ) -> u.Quantity:
        converter = self.__converters.get(to_)
        if converter is not None:
            return converter(value, **unit_kwargs)
        return value


def get_unit_applicators(group: h5py.Group, header: OpenCosmoHeader):
    convention = UnitConvention(header.file.unit_convention)
    transitions = get_unit_transitions(header)

    applicators = {}
    for name, column in group.items():
        base_unit = get_raw_units(column)
        applicators[name] = UnitApplicator(base_unit, convention, transitions)
    return applicators


def get_raw_units(column: h5py.Dataset):
    if "unit" in column.attrs:
        if (us := column.attrs["unit"]) == "None" or us == "":
            return None
        if (unit := KNOWN_UNITS.get(us)) is not None:
            return unit
        try:
            return u.Unit(us)
        except ValueError:
            return None

    return None
