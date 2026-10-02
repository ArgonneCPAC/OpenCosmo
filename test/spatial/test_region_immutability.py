from __future__ import annotations

import astropy.units as u  # type: ignore
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore
from opencosmo.spatial.region import BoxRegion, ConeRegion, HealpixRegion, SkyboxRegion


def test_box_region_copies_constructor_values_and_bounds() -> None:
    center = [1, 2, 3]
    halfwidths = [1, 1, 1]
    region = BoxRegion(center, halfwidths)  # type: ignore[arg-type]

    center[0] = 10
    halfwidths[0] = 10
    bounds = region.bounds
    bounds[0] = (10, 20)

    assert region.bounds == [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]


def test_healpix_region_copies_constructor_and_property_arrays() -> None:
    pixels = np.array([1, 2, 3], dtype=np.int64)
    region = HealpixRegion(pixels, 1)

    pixels[0] = 9
    returned = region.pixels
    returned[1] = 9

    assert np.array_equal(region.pixels, [1, 2, 3])


def test_healpix_region_copies_chunked_constructor_arrays() -> None:
    starts = np.array([1], dtype=np.int64)
    sizes = np.array([2], dtype=np.int64)
    region = HealpixRegion((starts, sizes), 1)

    starts[0] = 9
    sizes[0] = 9

    assert np.array_equal(region.pixels, [1, 2])


def test_cone_region_copies_astropy_inputs() -> None:
    center = SkyCoord(10 * u.deg, 20 * u.deg)
    radius = 5 * u.deg
    region = ConeRegion(center, radius)

    center.data.lon[...] = 30 * u.deg
    radius[...] = 10 * u.deg

    assert region.center.ra.deg == 10
    assert region.radius.to_value(u.deg) == 5


def test_skybox_region_copies_astropy_inputs() -> None:
    p1 = SkyCoord(10 * u.deg, 20 * u.deg)
    p2 = SkyCoord(30 * u.deg, 40 * u.deg)
    region = SkyboxRegion(p1, p2)

    p1.data.lon[...] = 50 * u.deg
    p2.data.lat[...] = 50 * u.deg

    assert region.ra_bounds == (10, 30)
    assert region.dec_bounds == (20, 40)
