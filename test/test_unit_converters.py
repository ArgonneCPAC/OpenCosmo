import astropy.units as u
import numpy as np
from astropy.cosmology import units as cu
from opencosmo.units.converters import (
    comoving_to_physical,
    get_unit_distance_power,
    get_unit_without_h,
    physical_to_comoving,
)


def test_unit_without_h_preserves_log_container():
    assert get_unit_without_h(u.DexUnit(u.Msun / cu.littleh)) == u.DexUnit(u.Msun)
    assert get_unit_without_h(u.MagUnit(u.Msun / cu.littleh)) == u.MagUnit(u.Msun)
    assert get_unit_without_h(u.DexUnit(u.erg / u.s)) == u.DexUnit(u.erg / u.s)


def test_distance_power_of_log_unit():
    assert get_unit_distance_power(u.DexUnit(u.Mpc / cu.littleh)) == 1
    assert get_unit_distance_power(u.DexUnit(u.Mpc**3)) == 3
    assert get_unit_distance_power(u.DexUnit(u.Msun)) is None


def test_comoving_physical_roundtrip_log_quantity():
    scale_factor = np.array([0.5, 0.25])
    linear = np.array([10.0, 100.0]) * u.Mpc**2
    logged = np.log10(linear.value) * u.DexUnit(u.Mpc**2)

    physical = comoving_to_physical(logged, scale_factor)
    assert physical.unit == u.DexUnit(u.Mpc**2)
    expected = comoving_to_physical(linear, scale_factor)
    assert np.allclose(physical.value, np.log10(expected.value))

    comoving = physical_to_comoving(physical, scale_factor, u.DexUnit(u.Mpc**2))
    assert np.allclose(comoving.value, logged.value)
