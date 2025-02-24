import astropy.cosmology.units as cu
import astropy.units as u
from astropy.table import Column


def get_column_parsers():
    return [parse_velocities, parse_mass]


def parse_velocities(column: Column):
    VELOCITY_COLUMNS = ["VX", "VY", "VZ"]
    name = column.name.upper()
    if name[-2:] in VELOCITY_COLUMNS:
        return column * u.km / u.s


def parse_mass(column: Column):
    column_name_parts = column.name.split("_")

    def is_mass_column(part):
        return part == "mass" or part[0] == "M"

    if any(is_mass_column(part) for part in column_name_parts):
        return column * u.Msun / cu.littleh
