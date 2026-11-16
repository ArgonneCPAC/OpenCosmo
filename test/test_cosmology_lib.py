from __future__ import annotations

import astropy.cosmology as ac
import astropy.units as u
import numpy as np
import pytest
from opencosmo._lib import cosmology as cosmolib
from opencosmo.collection.lightcone.plugins import redshift_from_chi

COSMOLOGIES = {
    "flat_lcdm": ac.FlatLambdaCDM(H0=67.66, Om0=0.3097, Tcmb0=0.0),
    "flat_lcdm_radiation": ac.FlatLambdaCDM(
        H0=67.66, Om0=0.3097, Tcmb0=2.7255, m_nu=0.0 * u.eV
    ),
    "open_lcdm": ac.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.6, Tcmb0=0.0),
    "flat_wcdm": ac.FlatwCDM(H0=70.0, Om0=0.3, w0=-0.9, Tcmb0=0.0),
    "w0wacdm": ac.w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, wa=0.2, Tcmb0=0.0),
}

REDSHIFTS = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])


def to_lib(cosmology: ac.FLRW):
    return cosmolib.make_cosmology(
        h=cosmology.h,
        Om0=cosmology.Om0,
        Ode0=cosmology.Ode0,
        Ok0=cosmology.Ok0,
        Or0=cosmology.Ogamma0 + cosmology.Onu0,
        w0=getattr(cosmology, "w0", -1.0),
        wa=getattr(cosmology, "wa", 0.0),
    )


@pytest.fixture(params=sorted(COSMOLOGIES))
def cosmologies(request):
    astropy_cosmology = COSMOLOGIES[request.param]
    return astropy_cosmology, to_lib(astropy_cosmology)


def test_comoving_distance_matches_astropy(cosmologies):
    astropy_cosmology, cosmology = cosmologies
    expected = astropy_cosmology.comoving_distance(REDSHIFTS).to_value(u.Mpc)

    scalar = np.array([cosmology.comoving_distance(z, None) for z in REDSHIFTS])
    vectorized = cosmology.comoving_distances(REDSHIFTS)

    assert np.allclose(scalar, expected, rtol=1e-5, atol=1e-3)
    assert np.allclose(vectorized, expected, rtol=1e-5, atol=1e-3)


def test_redshift_at_distance_matches_astropy(cosmologies):
    astropy_cosmology, cosmology = cosmologies
    redshifts = REDSHIFTS[REDSHIFTS > 0]
    distances = astropy_cosmology.comoving_distance(redshifts)

    expected = np.array(
        [
            d.to_value(
                ac.units.redshift,
                ac.units.redshift_distance(astropy_cosmology, kind="comoving"),
            )
            for d in distances
        ]
    )
    result = np.array(
        [cosmology.redshift_at_distance(d) for d in distances.to_value(u.Mpc)]
    )

    assert np.allclose(result, expected, rtol=1e-7)
    assert np.allclose(result, redshifts, rtol=1e-7)


def test_default_parameters_match_flat_lcdm():
    astropy_cosmology = COSMOLOGIES["flat_lcdm"]
    cosmology = cosmolib.make_cosmology(
        h=astropy_cosmology.h, Om0=astropy_cosmology.Om0
    )
    expected = astropy_cosmology.comoving_distance(REDSHIFTS).to_value(u.Mpc)
    assert np.allclose(
        cosmology.comoving_distances(REDSHIFTS), expected, rtol=1e-5, atol=1e-3
    )


def test_comoving_distance_grid_matches_astropy(cosmologies):
    astropy_cosmology, cosmology = cosmologies
    z_grid, chi_grid = cosmology.comoving_distance_grid(0.05, 3.0, 4096)

    assert np.array_equal(z_grid, np.linspace(0.05, 3.0, 4096))
    expected = astropy_cosmology.comoving_distance(z_grid).to_value(u.Mpc)
    assert np.allclose(chi_grid, expected, rtol=1e-5)


def test_redshift_from_chi_matches_astropy(cosmologies):
    astropy_cosmology, _ = cosmologies
    redshifts = np.linspace(0.01, 3.0, 1000)
    chi = astropy_cosmology.comoving_distance(redshifts)

    result = redshift_from_chi(chi, astropy_cosmology)["redshift"]
    assert np.allclose(result.value, redshifts, rtol=1e-5)


@pytest.mark.parametrize("distance", [-1.0, np.inf, np.nan, 1.0e6])
def test_redshift_at_distance_rejects_invalid_distance(distance):
    cosmology = to_lib(COSMOLOGIES["flat_lcdm"])
    with pytest.raises(ValueError):
        cosmology.redshift_at_distance(distance)


@pytest.mark.parametrize(
    ("z_min", "z_max", "n"), [(1.0, 0.5, 10), (-0.1, 1.0, 10), (0.0, 1.0, 1)]
)
def test_comoving_distance_grid_rejects_invalid_input(z_min, z_max, n):
    cosmology = to_lib(COSMOLOGIES["flat_lcdm"])
    with pytest.raises(ValueError):
        cosmology.comoving_distance_grid(z_min, z_max, n)
