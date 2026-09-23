import healpy as hp
import numpy as np
from astropy.io import fits

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
