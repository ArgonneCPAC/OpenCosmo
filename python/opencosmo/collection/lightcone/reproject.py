import healpy as hp
import numpy as np
from astropy.wcs import WCS


def reproject_to_cartesian(
    pixels: np.ndarray,
    values: dict[str, np.ndarray],
    nside: int,
    center: tuple[float, float],
    width: float,
    resolution: int,
):
    from time import time

    start = time()
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [center[0], center[1]]
    wcs.wcs.crpix = [(resolution + 1) / 2, (resolution + 1) / 2]
    wcs.wcs.cdelt = [-width / resolution, width / resolution]

    ys, xs = np.indices((resolution, resolution))
    lon, lat = wcs.pixel_to_world_values(xs, ys)  # degrees, in the WCS frame
    # If the map frame differs from the WCS frame, convert here with astropy SkyCoord.
    theta, phi = np.radians(90 - lat), np.radians(lon)

    pix, wts = hp.get_interp_weights(nside, theta.ravel(), phi.ravel(), nest=True)

    uniq, inv = np.unique(pix, return_inverse=True)  # already sorted

    _, idxi, idxo = np.intersect1d(pixels, uniq, return_indices=True)

    output = {}
    inv = inv.reshape(pix.shape)
    for name, arr in values.items():
        output_arr = np.full(len(uniq), fill_value=np.nan, dtype=np.float64)
        output_arr[idxo] = arr[idxi]
        output[output == hp.UNSEEN] = np.nan
        output[name] = (
            (output_arr[inv] * wts).sum(axis=0).reshape(resolution, resolution)
        )
    end = time()
    print(round(end - start, 3))
    return output
