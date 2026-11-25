from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.cosmology import units as cu

from opencosmo.units import UnitConvention
from opencosmo.units.known import get_redshift_column, get_scale_factor_column

if TYPE_CHECKING:
    from collections.abc import Callable
    from fractions import Fraction

    from opencosmo.dataset.state import DatasetState
    from opencosmo.dtypes.cosmology import FlrwCosmology
    from opencosmo.units.types import AnyUnit

type Converter = Callable[..., u.Quantity]
type Transitions = dict[UnitConvention, Converter]


def get_unit_transitions(
    unit: AnyUnit,
    base_convention: UnitConvention,
    cosmology: FlrwCosmology,
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


def get_comoving_transitions(
    unit: AnyUnit, cosmology: FlrwCosmology, is_comoving: bool
):
    error = partial(
        raise_convert_error, from_=UnitConvention.COMOVING, to_=UnitConvention.SCALEFREE
    )

    transitions: Transitions = {UnitConvention.SCALEFREE: error}
    inv_transitions: Transitions = {UnitConvention.SCALEFREE: error}
    distance_power = get_unit_distance_power(unit)
    if distance_power is not None and is_comoving:
        transitions[UnitConvention.PHYSICAL] = comoving_to_physical
        inv_transitions[UnitConvention.PHYSICAL] = partial(
            physical_to_comoving,
            base_unit=unit,
        )
    units = {UnitConvention.COMOVING: unit, UnitConvention.PHYSICAL: unit}
    return transitions, inv_transitions, units


def get_scalefree_transitions(
    unit: AnyUnit, cosmology: FlrwCosmology, is_comoving: bool
):
    hless_unit = get_unit_without_h(unit)
    transitions: Transitions = {}
    inv_transitions: Transitions = {}

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
def get_unit_without_h(unit: AnyUnit) -> AnyUnit:
    u_base = unit.physical_unit if isinstance(unit, u.LogUnit) else unit
    if not isinstance(u_base, u.UnitBase):
        return unit

    try:
        index = u_base.bases.index(cu.littleh)
    except ValueError:
        return unit
    power = u_base.powers[index]
    hless = u_base / cu.littleh**power
    if isinstance(unit, u.LogUnit):
        # Re-wrap so log-valued data is not exponentiated by Quantity.to
        return type(unit)(hless, unit.function_unit)
    return hless


@cache
def get_unit_distance_power(unit: AnyUnit) -> float | Fraction | None:
    u_base = unit.physical_unit if isinstance(unit, u.LogUnit) else unit
    decomposed = u_base.decompose()
    if not isinstance(decomposed, u.UnitBase):
        return None
    try:
        index = decomposed.bases.index(u.m)
    except ValueError:
        return None
    return decomposed.powers[index]


def add_littleh(
    value: u.Quantity, cosmology: FlrwCosmology, new_unit: AnyUnit, **kwargs
):
    return value.to(
        new_unit,
        cu.with_H0(cosmology.H0),
    )


def physical_to_comoving(
    value: u.Quantity, scale_factor: float | np.ndarray, base_unit: AnyUnit, **kwargs
):
    power = get_unit_distance_power(base_unit)
    if power is not None:
        return __scale(value, scale_factor ** (-power))
    return value


def remove_littleh(value: u.Quantity, cosmology: FlrwCosmology, **kwargs) -> u.Quantity:
    """
    Remove little h from the units of the input table. For comoving
    coordinates, this is the second step after parsing the units themselves.
    """
    new_unit = get_unit_without_h(value.unit)
    if new_unit != value.unit:
        return value.to(
            new_unit,
            cu.with_H0(cosmology.H0),
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
        return __scale(value, scale_factor**power)
    return value


def __scale(value: u.Quantity, factor: float | np.ndarray) -> u.Quantity:
    # Scaling the physical value of a log quantity is an addition in log space
    if isinstance(value.unit, u.LogUnit):
        log_factor = u.Dex(np.log10(u.Quantity(factor, u.one).value))
        return value + log_factor  # ty: ignore[invalid-explicit-return]  # astropy has no stubs; Quantity + Quantity is a Quantity
    return value * factor


def get_scale_factor(dataset: DatasetState, cosmology, redshift):
    from opencosmo.dataset import operations as dsops

    baseline = dsops.with_units(dataset, dataset.unit_handler.base_convention.value, {})
    if (a_column := get_scale_factor_column(baseline.columns)) is not None:
        return dsops.get_data(
            dsops.select(baseline, a_column), "numpy", wrap_single=True
        )[a_column]

    if (z_column := get_redshift_column(baseline.columns)) is not None:
        z = dsops.get_data(dsops.select(baseline, z_column), "numpy", wrap_single=True)[
            z_column
        ]
        return 1 / (1 + z)

    return cosmology.scale_factor(redshift)


def scalefree_to_physical(
    value: u.Quantity,
    scale_factor: float | np.ndarray,
    cosmology: FlrwCosmology,
):
    new_value = remove_littleh(value, cosmology)
    return comoving_to_physical(new_value, scale_factor)


def physical_to_scalefree(
    value: u.Quantity,
    scale_factor: float | np.ndarray,
    base_unit: AnyUnit,
    cosmology: FlrwCosmology,
):
    new_value = physical_to_comoving(value, scale_factor, base_unit)
    return add_littleh(new_value, cosmology, base_unit)


def raise_convert_error(*args, from_: UnitConvention, to_: UnitConvention, **kwargs):
    raise ValueError(
        f"Units in convention {from_!s} cannot be converted to units in convention {to_!s}"
    )
