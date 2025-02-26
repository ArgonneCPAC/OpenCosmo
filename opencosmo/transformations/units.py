from typing import Optional

import astropy.cosmology.units as cu
import astropy.units as u
from astropy.table import Column, Table


def apply_units_by_name(table: Table) -> None:
    """
    InPlaceTableTransformation that applies units to the columns.
    """
    for colname in table.columns:
        column = table[colname]
        if column.unit is not None:
            continue
        unit = parse_column_name(colname)
        if unit is not None:
            column.unit = unit


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
