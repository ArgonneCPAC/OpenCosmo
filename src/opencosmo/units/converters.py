from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING, Optional

import astropy.units as u
from astropy.cosmology import units as cu

from opencosmo.units import UnitConvention

if TYPE_CHECKING:
    import numpy as np
    from astropy.cosmology import Cosmology

    from opencosmo.dataset.state import DatasetState

KNOWN_SCALEFACTOR_COLUMNS = {"fof_halo_center_a"}
KNOWN_REDSHIFT_COLUMNS = {"redshift", "redshift_true"}


def get_unit_transitions(
    unit: u.Unit,
    base_convention: UnitConvention,
    cosmology: Cosmology,
    is_comoving: bool,
):
    match base_convention:
        case UnitConvention.PHYSICAL | UnitConvention.UNITLESS:
            return {}, {}, {}
        case UnitConvention.SCALEFREE:
            return get_scalefree_transitions(unit, cosmology, is_comoving)
        case UnitConvention.COMOVING:
            return get_comoving_transitions(unit, cosmology, is_comoving)
        case _:
            raise ValueError(f"Invalid unit convention {base_convention}")


def get_comoving_transitions(unit: u.Unit, cosmology: Cosmology, is_comoving: bool):
    error = partial(
        raise_convert_error, from_=UnitConvention.COMOVING, to_=UnitConvention.SCALEFREE
    )

    transitions = {UnitConvention.SCALEFREE: error}
    inv_transitions = {UnitConvention.SCALEFREE: error}
    distance_power = get_unit_distance_power
    if distance_power is not None and is_comoving:
        transitions[UnitConvention.PHYSICAL] = comoving_to_physical  # type: ignore
        inv_transitions[UnitConvention.PHYSICAL] = partial(
            physical_to_comoving,
            base_unit=unit,
        )
    units = {UnitConvention.COMOVING: unit, UnitConvention.PHYSICAL: unit}
    return transitions, inv_transitions, units


def get_scalefree_transitions(unit: u.Unit, cosmology: Cosmology, is_comoving: bool):
    hless_unit = get_unit_without_h(unit)
    transitions = {}
    inv_transitions = {}

    if hless_unit != unit:
        rem_h = partial(remove_littleh, cosmology=cosmology)
        add_h = partial(add_littleh, cosmology=cosmology, new_unit=unit)
        transitions[UnitConvention.COMOVING] = rem_h
        inv_transitions[UnitConvention.COMOVING] = add_h

    distance_power = get_unit_distance_power(unit)
    if distance_power is not None and is_comoving:
        transitions[UnitConvention.PHYSICAL] = partial(
            scalefree_to_physical, cosmology=cosmology
        )
        inv_transitions[UnitConvention.PHYSICAL] = partial(
            physical_to_scalefree, base_unit=unit, cosmology=cosmology
        )
    elif transitions.get(UnitConvention.COMOVING) is not None:
        transitions[UnitConvention.PHYSICAL] = transitions[UnitConvention.COMOVING]
        inv_transitions[UnitConvention.PHYSICAL] = inv_transitions[
            UnitConvention.COMOVING
        ]

    units = {
        UnitConvention.SCALEFREE: unit,
        UnitConvention.COMOVING: hless_unit,
        UnitConvention.PHYSICAL: hless_unit,
    }

    return transitions, inv_transitions, units


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
    except (ValueError, AttributeError):
        return unit
    power = u_base.powers[index]
    new_unit = u_base / cu.littleh**power

    return new_unit


@cache
def get_unit_distance_power(unit: u.Unit) -> Optional[float]:
    decomposed = unit.decompose()
    try:
        index = decomposed.bases.index(u.m)
        return decomposed.powers[index]
    except (ValueError, AttributeError):
        return None


def add_littleh(value: u.Quantity, cosmology: Cosmology, new_unit: u.Unit, **kwargs):
    return value.to(
        new_unit,
        cu.with_H0(cosmology.H0),  # pyrefly: ignore[missing-attribute]
    )


def physical_to_comoving(
    value: u.Quantity, scale_factor: float | np.ndarray, base_unit: u.Unit, **kwargs
):
    power = get_unit_distance_power(base_unit)
    if power is not None:
        return value / (scale_factor**power)
    return value


def remove_littleh(value: u.Quantity, cosmology: Cosmology, **kwargs) -> u.Quantity:
    """
    Remove little h from the units of the input table. For comoving
    coordinates, this is the second step after parsing the units themselves.
    """
    new_unit = get_unit_without_h(value.unit)
    if new_unit != value.unit:
        return value.to(
            new_unit,
            cu.with_H0(cosmology.H0),  # pyrefly: ignore[missing-attribute]
        )
    return value


def comoving_to_physical(
    value: u.Quantity, scale_factor: float | np.ndarray, **kwargs
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


def get_scale_factor(dataset: "DatasetState", cosmology, redshift):
    columns = set(dataset.columns)
    for column in KNOWN_SCALEFACTOR_COLUMNS:
        if column in columns:
            col = dataset.select(column).get_data()[column]
            return col

    for column in KNOWN_REDSHIFT_COLUMNS:
        if column in columns:
            col = dataset.select(column).get_data()[column]
            return 1 / (1 + col)

    return cosmology.scale_factor(redshift)


def scalefree_to_physical(
    value: u.Quantity,
    scale_factor: float | np.ndarray,
    cosmology: Cosmology,
):
    new_value = remove_littleh(value, cosmology)
    return comoving_to_physical(new_value, scale_factor)


def physical_to_scalefree(
    value: u.Quantity,
    scale_factor: float | np.ndarray,
    base_unit: u.Quantity,
    cosmology: Cosmology,
):
    new_value = physical_to_comoving(value, scale_factor, base_unit)
    return add_littleh(new_value, cosmology, base_unit)


def raise_convert_error(*args, from_: UnitConvention, to_: UnitConvention, **kwargs):
    raise ValueError(
        f"Units in convention {str(from_)} cannot be converted to units in convention {str(to_)}"
    )
