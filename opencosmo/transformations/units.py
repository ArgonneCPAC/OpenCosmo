import astropy.cosmology.units as cu
import astropy.units as u
from astropy.table import Column, Table


def apply_units(table: Table) -> None:
    """
    InPlaceTableTransformation that applies units to the columns.

    where are monads when you need them :'(
    """
    parsers = get_column_parsers()
    for colname in table.columns:
        is_updated = False
        new_column = table[colname]
        for parser in parsers:
            updated_column = parser(new_column)
            if updated_column is not None:
                new_column = updated_column
                is_updated = True
        if is_updated:
            table[colname] = new_column


def get_column_parsers():
    return [parse_velocities, parse_mass]


def parse_velocities(column: Column):
    VELOCITY_COLUMNS = ["VX", "VY", "VZ"]
    name = column.name.upper()
    if name[-2:] in VELOCITY_COLUMNS:
        column.unit = u.km / u.s
        return column


def parse_mass(column: Column):
    column_name_parts = column.name.split("_")

    def is_mass_part(part):
        return part == "mass" or part[0] == "M"

    if any(is_mass_part(part) for part in column_name_parts):
        column.unit = u.Msun / cu.littleh
        return column
