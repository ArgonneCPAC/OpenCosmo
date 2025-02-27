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


def get_unit_transformation_generators(
    convention="comoving",
) -> list[t.TransformationGenerator]:
    """
    Get the unit transformation generators for a given convention.

    We use generators for units because it is most appropriate to think
    of units as fundamental to the data, even when they don't actually
    appear in the hdf5 file.

    HACC data by default using scale-free comoving units.
    """
    if convention is not None:
        return [
            generate_attribute_unit_transformation,
            generate_name_unit_transformation,
        ]


def get_unit_transformations(
    cosmology: Cosmology,
    convention: str = "comoving",
) -> dict[str, list[t.TableTransformation]]:
    """
    Get further transformations based on the requested unit convention.

    These always apply after the initial transformations generated above.
    """
    if convention == "comoving":
        remove_h = partial(remove_littleh, cosmology=cosmology)
        return {"table": [remove_h]}
    return {}


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
    """
    Check the attributes of an hdf5 dataset to see if information about units is stored
    there.

    The raw HACC data does not store units in this way, relying instead on standard
    naming conventions. However if a user creates a new column, we want to be able to
    store it in our standard format without losing unit information and we cannot rely
    on them following our naming conventions.
    """
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


def generate_name_unit_transformation(
    input: Dataset,
) -> dict[str, list[t.Transformation]]:
    """
    Generator for unit transformations based on the name of the column. Just
    checks for the HACC naming conventions.
    """
    name = input.name.split("/")[-1]
    unit = parse_column_name(name)
    if unit is not None:
        apply_func: t.Transformation = apply_unit(column_name=name, unit=unit)
        return {"column": [apply_func]}


class apply_unit:
    """
    Apply a unit to an input column. Ensuring that the correct column is
    passed will be the responsibility of the caller.

    Has to be a class so it can implement the ColumnTransformation protocol.
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
    An alternative option for applying units based on  name that is
    a table transformation rather than a column transformation. Not
    currently in use.

    The advantage of column transformations is that they are much more
    efficient if we are trying to produce a filter based on a single column
    in a very large table. We simply load the column, apply the filters that
    apply to it, and evaluate the filter.
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
