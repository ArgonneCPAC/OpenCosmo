import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def convert_format(
    format: str,
    region_pixels: np.ndarray,
    region_data: dict[str, np.ndarray],
    nside: int,
    center: tuple[float, float],
    angular_size: float,
    npix: int,
) -> fits.HDUList:
    if format == "hdul":
        reprojected_data = reproject_to_cartesian(
            region_pixels, region_data, nside, center, angular_size, npix
        )
        wcs = reprojected_data.pop("wcs")
        assert isinstance(wcs, WCS)
        header = wcs.to_header()
        extensions = [
            fits.ImageHDU(data=data, header=header, name=name)
            for name, data in reprojected_data.items()
        ]
        return fits.HDUList([fits.PrimaryHDU(header=header), *extensions])

    raise ValueError(f"Unsupported cutout format: {format!r}")


def reproject_to_cartesian(
    pixels: np.ndarray,
    values: dict[str, np.ndarray],
    nside: int,
    center: tuple[float, float],
    width: float,
    resolution: int,
) -> dict[str, np.ndarray | WCS]:

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

    output = {"wcs": wcs}
    inv = inv.reshape(pix.shape)
    for name, arr in values.items():
        output_arr = np.full(len(uniq), fill_value=np.nan, dtype=np.float64)
        selected = np.asarray(arr[idxi], dtype=np.float64)
        output_arr[idxo] = np.where(selected == hp.UNSEEN, np.nan, selected)
        output[name] = (
            (output_arr[inv] * wts).sum(axis=0).reshape(resolution, resolution)
        )

    return output
