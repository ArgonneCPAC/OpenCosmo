from functools import partial
from typing import Optional
from warnings import warn

import astropy.cosmology.units as cu  # type: ignore
import astropy.units as u  # type: ignore
from astropy.cosmology import Cosmology
from astropy.table import Column, Table  # type: ignore
from h5py import Dataset  # type: ignore

from opencosmo import transformations as t

_ = u.add_enabled_units(cu)


def get_unit_transformation_generators() -> list[t.TransformationGenerator]:
    return [generate_attribute_unit_transformation]


def get_unit_transformations(
    cosmology: Cosmology,
    convention: str = "comoving",
) -> dict[str, list[t.TableTransformation]]:
    if convention == "comoving":
        remove_h = partial(remove_littleh, cosmology=cosmology)
        return {"table": [apply_units_by_name, remove_h]}
    return {"table": [apply_units_by_name]}


def remove_littleh(input: Table, cosmology: Cosmology) -> Optional[Table]:
    """
    Remove little h from the units of the input table. For comoving
    coordinates, this is the second step after parsing the units themselves.
    """
    table = input
    for column in table.columns:
        if (unit := table[column].unit) is not None:
            try:
                index = unit.bases.index(cu.littleh)
            except ValueError:
                continue
            power = unit.powers[index]
            new_unit = unit / cu.littleh**power
            table[column] = table[column].to(new_unit, cu.with_H0(cosmology.H0))
    return table


def generate_attribute_unit_transformation(
    input: Dataset,
) -> dict[str, list[t.Transformation]]:
    if "unit" in input.attrs:
        try:
            unit = u.Unit(input.attrs["unit"])
        except ValueError:
            warn(
                f"Invalid unit {input.attrs['unit']} in column {input.name}. "
                "Values will be unitless..."
            )
            return {}
        apply_func: t.Transformation = apply_unit(
            column_name=input.name.split("/")[-1], unit=unit
        )
        return {"column": [apply_func]}
    return {}


class apply_unit:
    """
    Apply a unit to an input column. Ensuring that the correct column is
    passed will be the responsibility of the caller.
    """

    def __init__(self, column_name: str, unit: u.Unit):
        self.__name = column_name
        self.unit = unit

    def __call__(self, input: Column) -> Optional[Column]:
        if input.unit is None:
            input.unit = self.unit
        return input

    @property
    def column_name(self) -> str:
        return self.__name


def apply_units_by_name(input: Table) -> Optional[Table]:
    """
    Apply units to columns based on their names. The column names in
    HACC are generally pretty regular, so we can parse them pretty
    easily. For derived columns, they will be attached as an attribute
    on the hdf5 dataset.
    """
    modified = False
    table = input
    for colname in table.columns:
        column = table[colname]
        if column.unit is not None:
            continue
        unit = parse_column_name(colname)
        if unit is not None:
            column.unit = unit
            modified = True
    if modified:
        return table
    return None


def parse_column_name(column_name: str) -> Optional[u.Quantity | u.Unit]:
    name_parts = column_name.split("_")
    if len(name_parts) == 1:
        return parse_single_name(name_parts[0])
    else:
        return parse_multipart_name(name_parts)


def parse_single_name(name: str) -> Optional[u.Quantity | u.Unit]:
    match name:
        case "x" | "y" | "z" | "hh":
            return u.Mpc / cu.littleh
        case "phi" | "uu":
            return (u.km / u.s) ** 2
        case "mass" | "mbh":
            return u.Msun / cu.littleh
        case "rho":
            return (u.Msun / u.kpc**3) * cu.littleh**2
        case "bhr":
            return u.Msun / u.yr
        case "age":  # Fix needed:
            return None
        case _:
            return None


def parse_multipart_name(name: list[str]) -> Optional[u.Quantity | u.Unit]:
    if "mass" in name or name[-1][0].upper() == "M":
        return u.Msun / cu.littleh
    if "radius" in name or name[-1][0] == "R" or name[-1].startswith("rad"):
        return u.Mpc / cu.littleh
    elif "vel" in name or name[-1] in ["vx", "vy", "vz"]:
        return u.km / u.s
    elif name[-1] in ["x", "y", "z"]:
        return u.Mpc / cu.littleh
    elif name[-1][0] == "Y":
        return (u.Mpc / cu.littleh) ** 2
    elif name[-1][0] == "T":
        return u.Kelvin
    elif name[-1][0] == "L":
        return u.DexUnit(u.erg / u.s)
    elif name[-1] == "entropy":
        return u.kev / u.cm**2
    elif name[-1] == "ne":
        return u.cm**-3
    elif name[-1][0] == "t":
        return u.Gyr
    elif name[-1] in ["sfr", "bhr"]:
        return u.Msun / u.yr
    elif name[-1] == "ke":
        return (u.km / u.s) ** 2
    return None
