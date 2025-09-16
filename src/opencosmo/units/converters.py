from functools import cache, partial
from typing import Optional

import astropy.units as u
from astropy.cosmology import Cosmology
from astropy.cosmology import units as cu
from numpy.typing import ArrayLike

from opencosmo.header import OpenCosmoHeader
from opencosmo.units import UnitConvention


def get_unit_transitions(header: OpenCosmoHeader):
    convention = UnitConvention(header.file.unit_convention)
    remove_h = partial(remove_littleh, cosmology=header.cosmology)
    sf_to_phys = partial(scalefree_to_physical, cosmology=header.cosmology)
    cm_to_phys = partial(comoving_to_physical, cosmology=header.cosmology)

    match convention:
        case (UnitConvention.PHYSICAL, UnitConvention.UNITLESS):
            return {}
        case UnitConvention.SCALEFREE:
            return {
                UnitConvention.COMOVING: remove_h,
                UnitConvention.PHYSICAL: sf_to_phys,
            }
        case UnitConvention.COMOVING:
            return {UnitConvention.PHYSICAL: cm_to_phys}


@cache
def get_unit_without_h(unit: u.Unit) -> u.Unit:
    try:
        if isinstance(unit, u.DexUnit):
            u_base = unit.physical_unit
            constructor = u.DexUnit
        else:
            u_base = unit

            def constructor(x):
                return x

    except AttributeError:
        return unit

    try:
        index = u_base.bases.index(cu.littleh)
    except ValueError:
        return unit
    power = u_base.powers[index]
    new_unit = constructor(u_base / cu.littleh**power)
    return new_unit


@cache
def get_unit_distance_power(unit: u.Unit) -> Optional[float]:
    decomposed = unit.decompose()
    try:
        index = decomposed.bases.index(u.m)
        return decomposed.powers[index]
    except (ValueError, AttributeError):
        return None


def remove_littleh(value: u.Quantity, cosmology: Cosmology, **kwargs) -> u.Quantity:
    """
    Remove little h from the units of the input table. For comoving
    coordinates, this is the second step after parsing the units themselves.
    """
    new_unit = get_unit_without_h(value.unit)
    if new_unit != value.unit:
        return value.to(new_unit, cu.with_H0(cosmology.H0))
    return value


def comoving_to_physical(
    value: u.Quantity, scale_factor: ArrayLike, **kwargs
) -> u.Quantity:
    """
    Convert comoving coordinates to physical coordinates. This is the
    second step after parsing the units themselves.
    """

    unit = value.unit
    # Check if the units have distances in them
    power = get_unit_distance_power(unit)
    # multiply by the scale factor to the same power as the distance
    if power is not None:
        return value * scale_factor**power
    return value


def scalefree_to_physical(
    value: u.Quantity, cosmology: Cosmology, scale_factor: ArrayLike
):
    new_value = remove_littleh(value, cosmology)
    return comoving_to_physical(new_value, scale_factor)
