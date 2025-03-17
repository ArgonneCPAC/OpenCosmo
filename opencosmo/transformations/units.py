from enum import Enum
from functools import partial
from typing import Optional
from warnings import warn
from functools import reduce

import astropy.cosmology.units as cu  # type: ignore
import astropy.units as u  # type: ignore
from astropy.cosmology import Cosmology
from astropy.table import Column, Table  # type: ignore
from h5py import Dataset  as h5Dataset# type: ignore

from opencosmo import transformations as t
from opencosmo.header import OpenCosmoHeader

_ = u.add_enabled_units(cu)


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
        if (us := input.attrs["unit"]) == "None":
            return {}
        
        comoving = us.startswith("comoving ")
        
        if comoving:
            us = us.removeprefix("comoving ")

        # Check if there are multiplied factors
        units = us.split("*")
        units = [u_ if "^" in u_ or "log" in u_ else u_.strip("() ") for u_ in units]
        # handle carots
        powers = [1 if "^" not in u_ else int(u_.split("^")[-1]) for u_ in units]
        units = [u_.split("^")[0].strip("() ") if "^" in u_ else u_ for u_ in units]
        # handle logarithmic units
        units = [u_.replace("log10", "dex") for u_ in units]



        try:
            unit = reduce(lambda x, y: x * y, [u.Unit(u_)**p for u_, p in zip(units, powers)])
        except ValueError:
            print(units)
            warn(
                f"Invalid unit {us} in column {input.name}. "
                "Values will be unitless..."
            )
            return {}
        # astropy parses h as hours, so convert to littleh
        try:
            h_index = unit.bases.index(u.hour)
            power = unit.powers[h_index]
            unit = unit * cu.littleh**power / u.hour**power
        except (ValueError, AttributeError):
            pass
            

        apply_func: t.Transformation = apply_unit(
            column_name=input.name.split("/")[-1], unit=unit
        )
        return {t.TransformationType.COLUMN: [apply_func]}
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
            input.unit = self.unit
        return input

    @property
    def column_name(self) -> str:
        return self.__name


