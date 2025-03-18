from enum import Enum
from functools import partial
from typing import Optional
from warnings import warn

import astropy.cosmology.units as cu  # type: ignore
import astropy.units as u  # type: ignore
from astropy.constants import m_p  # type: ignore
from astropy.cosmology import Cosmology
from astropy.table import Column, Table  # type: ignore
from h5py import Dataset as h5Dataset
from h5py import File, Group  # type: ignore

from opencosmo import transformations as t
from opencosmo.dataset.column import get_column_builders
from opencosmo.header import OpenCosmoHeader

_ = u.add_enabled_units(cu)

UNIT_MAP = {
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


class UnitConvention(Enum):
    COMOVING = "comoving"
    PHYSICAL = "physical"
    SCALEFREE = "scalefree"
    UNITLESS = "unitless"


def get_unit_transformation_generators() -> list[t.TransformationGenerator]:
    """
    Get the unit transformation generato.

    We use generators for units because it is most appropriate to think
    of units as fundamental to the data, even when they don't actually
    appear in the hdf5 file.

    Even if the user requests unitless data, we still need to have
    access to these so that we can apply units if they
    call Dataset.with_convention later.
    """
    return [
        generate_attribute_unit_transformations,
    ]


def get_unit_transition_transformations(
    convention: str, unit_transformations: t.TransformationDict, cosmology: Cosmology
) -> t.TransformationDict:
    """
    Given a dataset, the user can request a transformation to a different unit
    convention. The returns a new set of transformations that will take the
    dataset to the requested unit convention.
    """
    units = UnitConvention(convention)
    remove_h: t.TableTransformation = partial(remove_littleh, cosmology=cosmology)
    comoving_to_phys: t.TableTransformation = partial(
        comoving_to_physical, cosmology=cosmology, redshift=0
    )
    match units:
        case UnitConvention.COMOVING:
            update_transformations = {t.TransformationType.ALL_COLUMNS: [remove_h]}
        case UnitConvention.PHYSICAL:
            update_transformations = {
                t.TransformationType.ALL_COLUMNS: [remove_h, comoving_to_phys]
            }
            raise NotImplementedError("Physical units not yet implemented")
        case UnitConvention.SCALEFREE:
            update_transformations = {}
        case UnitConvention.UNITLESS:
            return {}

    for ttype in unit_transformations:
        existing = update_transformations.get(ttype, [])
        update_transformations[ttype] = unit_transformations[ttype] + existing
    return update_transformations


def get_default_unit_transformations(file: File | Group, header: OpenCosmoHeader):
    base_unit_transformations = get_base_unit_transformations(file["data"], header)
    to_comoving_transformations = get_unit_transition_transformations(
        "comoving", base_unit_transformations, header.cosmology
    )

    column_names = list(str(col) for col in file["data"].keys())
    builders = get_column_builders(to_comoving_transformations, column_names)

    return builders, base_unit_transformations


def get_base_unit_transformations(
    input: h5Dataset,
    header: OpenCosmoHeader,
) -> t.TransformationDict:
    """
    Get the base unit transformations for a given dataset. These transformations
    produce the units that the data are actually stored in. Datasets alwyas
    hold onto a copy of these transformations even if the user later requests
    a different unit convention.


    These always apply after the initial transformations generated above.
    """
    generators = get_unit_transformation_generators()
    base_transformations = t.generate_transformations(input, generators, {})
    return base_transformations


def remove_littleh(column: Column, cosmology: Cosmology) -> Optional[Table]:
    """
    Remove little h from the units of the input table. For comoving
    coordinates, this is the second step after parsing the units themselves.
    """
    if (unit := column.unit) is not None:
        # Handle dex units
        try:
            if isinstance(unit, u.DexUnit):
                u_base = unit.physical_unit
                constructor = u.DexUnit
            else:
                u_base = unit

                def constructor(x):
                    return x
        except AttributeError:
            return None

        try:
            index = u_base.bases.index(cu.littleh)
        except ValueError:
            return None
        power = u_base.powers[index]
        new_unit = constructor(u_base / cu.littleh**power)
        column = column.to(new_unit, cu.with_H0(cosmology.H0))
    return column


def comoving_to_physical(
    column: Column, cosmology: Cosmology, redshift: float
) -> Optional[Table]:
    """
    Convert comoving coordinates to physical coordinates. This is the
    second step after parsing the units themselves.
    """
    if (unit := column.unit) is not None:
        # Check if the units have distances in them
        decomposed = unit.decompose()
        try:
            index = decomposed.bases.index(u.m)
        except ValueError:
            return None
        power = decomposed.powers[index]
        # multiply by the scale factor to the same power as the distance
        a = 1 / (1 + redshift)
        column = column * a**power
        # Remove references to "com" from the name
        name_parts = column.name.split("_")
        if "com" in name_parts:
            name_parts.remove("com")
            column.name = "_".join(name_parts)

    return column


def generate_attribute_unit_transformations(
    input: h5Dataset,
) -> t.TransformationDict:
    """
    Check the attributes of an hdf5 dataset to see if information about units is stored
    there.

    The raw HACC data does not store units in this way, relying instead on standard
    naming conventions. However if a user creates a new column, we want to be able to
    store it in our standard format without losing unit information and we cannot rely
    on them following our naming conventions.
    """
    if "unit" in input.attrs:
        if (us := input.attrs["unit"]) == "None" or us == "":
            return {}
        try:
            unit = UNIT_MAP[us]
            apply_func: t.Transformation = apply_unit(
                column_name=input.name.split("/")[-1], unit=unit
            )
            return {t.TransformationType.COLUMN: [apply_func]}
        except KeyError:
            warn(
                f"Invalid unit {us} in column {input.name}. Values will be unitless..."
            )
            return {}
    return {}


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
            return input * self.unit
        return input

    @property
    def column_name(self) -> str:
        return self.__name
