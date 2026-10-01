import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from opencosmo.collection.lightcone.cutout import get_included_pixels


@pytest.mark.parametrize(
    "center",
    [
        SkyCoord(123.4, -20.0, unit="deg"),
        SkyCoord(359.9, 5.0, unit="deg"),
        SkyCoord(42.0, 88.0, unit="deg"),
    ],
)
def test_included_pixels_use_padded_cutout_radius(center):
    nside = 64
    size = 1.0
    radius = np.deg2rad(size) * np.sqrt(2.0) / 2.0 + 2.0 * hp.max_pixrad(nside)
    expected = hp.query_disc(
        nside,
        hp.ang2vec(center.ra.deg, center.dec.deg, lonlat=True),
        radius,
        nest=True,
    )

    pixels = get_included_pixels(SkyCoord([center]), size, nside)

    assert np.array_equal(pixels, np.sort(expected))


def test_included_pixels_union_multiple_centers():
    nside = 64
    size = 1.0
    centers = SkyCoord([10.0, 210.0], [-30.0, 45.0], unit="deg")
    expected = np.union1d(
        get_included_pixels(centers[:1], size, nside),
        get_included_pixels(centers[1:], size, nside),
    )

    assert np.array_equal(get_included_pixels(centers, size, nside), expected)


@pytest.mark.parametrize("size", [1 * u.deg, 60 * u.arcmin, np.pi / 180 * u.rad])
def test_included_pixels_accept_angular_quantities(size):
    center = SkyCoord(123.4, -20.0, unit="deg")

    expected = get_included_pixels(SkyCoord([center]), 1.0, 64)

    assert np.array_equal(get_included_pixels(SkyCoord([center]), size, 64), expected)


def test_included_pixels_treat_dimensionless_quantity_as_degrees():
    center = SkyCoord(123.4, -20.0, unit="deg")

    expected = get_included_pixels(SkyCoord([center]), 1.0, 64)

    assert np.array_equal(
        get_included_pixels(SkyCoord([center]), 1 * u.dimensionless_unscaled, 64),
        expected,
    )
