import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from opencosmo.collection.lightcone.healpix_map import HealpixMap
from opencosmo.collection.lightcone.lightcone import Lightcone
from opencosmo.collection.lightcone.reproject import make_hdulist


def test_make_hdulist_packs_reprojected_data():
    pixels = np.arange(hp.nside2npix(1), dtype=np.int64)
    values = {"signal": np.arange(len(pixels), dtype=np.float64)}

    cutout = make_hdulist(pixels, values, 1, (0.0, 0.0), 1.0, 4)

    assert isinstance(cutout, fits.HDUList)
    assert len(cutout) == 2
    assert cutout[1].name == "SIGNAL"
    assert cutout[1].data.shape == (4, 4)
    assert cutout[1].header["CTYPE1"] == "RA---TAN"
    assert cutout[1].header["CTYPE2"] == "DEC--TAN"


def test_make_hdulist_replaces_healpix_unseen_values_with_nan():
    pixels = np.arange(hp.nside2npix(1), dtype=np.int64)
    values = {"signal": np.full(len(pixels), hp.UNSEEN)}

    cutout = make_hdulist(pixels, values, 1, (0.0, 0.0), 1.0, 4)

    assert np.isnan(cutout[1].data).all()


@pytest.mark.parametrize("npix", [0, -1])
def test_map_cutouts_reject_nonpositive_npix(npix):
    healpix_map = HealpixMap.__new__(HealpixMap)
    centers = SkyCoord([], [], unit="deg")

    with pytest.raises(ValueError, match="npix must be positive"):
        next(healpix_map.cutouts(centers, 1.0, npix=npix))


@pytest.mark.parametrize("npix", [1.5, True, None])
def test_map_cutouts_reject_noninteger_npix(npix):
    healpix_map = HealpixMap.__new__(HealpixMap)
    centers = SkyCoord([], [], unit="deg")

    with pytest.raises(TypeError, match="npix must be an integer"):
        next(healpix_map.cutouts(centers, 1.0, npix=npix))


@pytest.mark.parametrize("size", [0, -1, np.nan, np.inf])
def test_map_cutouts_reject_invalid_size(size):
    healpix_map = HealpixMap.__new__(HealpixMap)
    centers = SkyCoord([], [], unit="deg")

    with pytest.raises(ValueError, match="positive finite value"):
        next(healpix_map.cutouts(centers, size))


@pytest.mark.parametrize("size", ["1", True, None, [1]])
def test_map_cutouts_require_numeric_or_quantity_size(size):
    healpix_map = HealpixMap.__new__(HealpixMap)
    centers = SkyCoord([], [], unit="deg")

    with pytest.raises(TypeError, match="numeric value or angular quantity"):
        next(healpix_map.cutouts(centers, size))


@pytest.mark.parametrize("size", [[1, 2] * u.deg, np.array([1, 2]) * u.deg])
def test_map_cutouts_require_scalar_size(size):
    healpix_map = HealpixMap.__new__(HealpixMap)
    centers = SkyCoord([], [], unit="deg")

    with pytest.raises(TypeError, match="Cutout size must be a scalar"):
        next(healpix_map.cutouts(centers, size))


def test_map_cutouts_reject_nonangular_quantity_size():
    healpix_map = HealpixMap.__new__(HealpixMap)
    centers = SkyCoord([], [], unit="deg")

    with pytest.raises(ValueError, match="Cutout size must have angular units"):
        next(healpix_map.cutouts(centers, 1 * u.m))


def test_lightcone_cutouts_require_companion_map():
    lightcone = Lightcone.__new__(Lightcone)
    lightcone._Lightcone__maps = None

    with pytest.raises(ValueError, match="No map was opened with this lightcone"):
        next(lightcone.cutouts(size=1.0))


def test_lightcone_cutouts_require_coordinate_columns():
    lightcone = Lightcone.__new__(Lightcone)
    lightcone._Lightcone__maps = HealpixMap.__new__(HealpixMap)
    lightcone[0] = type("DatasetColumns", (), {"columns": ["ra"]})()
    lightcone._Lightcone__hidden = set()
    lightcone._Lightcone__scope = type("EmptyScope", (), {"names": lambda self: []})()

    with pytest.raises(ValueError, match="require ra and dec columns; missing: dec"):
        next(lightcone.cutouts(size=1.0))
