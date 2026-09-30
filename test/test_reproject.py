import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from opencosmo.collection.lightcone.lightcone import Lightcone
from opencosmo.collection.lightcone.healpix_map import HealpixMap
from opencosmo.collection.lightcone.reproject import convert_format


def test_convert_format_packs_reprojected_data_into_hdulist():
    pixels = np.arange(hp.nside2npix(1), dtype=np.int64)
    values = {"signal": np.arange(len(pixels), dtype=np.float64)}

    cutout = convert_format("hdul", pixels, values, 1, (0.0, 0.0), 1.0, 4)

    assert isinstance(cutout, fits.HDUList)
    assert len(cutout) == 2
    assert cutout[1].name == "SIGNAL"
    assert cutout[1].data.shape == (4, 4)
    assert cutout[1].header["CTYPE1"] == "RA---TAN"
    assert cutout[1].header["CTYPE2"] == "DEC--TAN"


def test_convert_format_replaces_healpix_unseen_values_with_nan():
    pixels = np.arange(hp.nside2npix(1), dtype=np.int64)
    values = {"signal": np.full(len(pixels), hp.UNSEEN)}

    cutout = convert_format("hdul", pixels, values, 1, (0.0, 0.0), 1.0, 4)

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


def test_lightcone_cutouts_require_companion_map():
    lightcone = Lightcone.__new__(Lightcone)
    lightcone._Lightcone__maps = None

    with pytest.raises(ValueError, match="No map was opened with this lightcone"):
        next(lightcone.cutouts(size=1.0))
