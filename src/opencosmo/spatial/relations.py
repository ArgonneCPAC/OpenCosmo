from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from opencosmo.spatial.region import (
        BoxRegion,
        ConeRegion,
        HealpixRegion,
        SkyboxRegion,
    )


# ---------------------------------------------------------------------------
# 3D helpers
# ---------------------------------------------------------------------------


def __box_contains_box(region: BoxRegion, other: BoxRegion) -> bool:
    for b1, b2 in zip(region.bounds, other.bounds):
        if b1[0] > b2[0] or b1[1] < b2[1]:
            return False
    return True


def __box_contains_points(region: BoxRegion, coords: NDArray) -> NDArray:
    if coords.shape[0] != 3:
        raise ValueError("Expected a coordinate array with shape (3, n_points)!")
    mask = np.ones(coords.shape[1], dtype=bool)
    for bound, col in zip(region.bounds, coords):
        mask &= (col > bound[0]) & (col < bound[1])
    return mask


def __box_intersects_box(region: BoxRegion, other: BoxRegion) -> bool:
    for b1, b2 in zip(region.bounds, other.bounds):
        if b1[0] > b2[1] or b1[1] < b2[0]:
            return False
    return True


# ---------------------------------------------------------------------------
# 2D helpers
# ---------------------------------------------------------------------------


def __cone_contains_cone(region: ConeRegion, other: ConeRegion) -> bool:
    dtheta = region.center.separation(other.center)
    return region.radius > (dtheta + other.radius)


def __cone_contains_point(region: ConeRegion, coords: SkyCoord) -> NDArray:
    seps = region.center.separation(coords)
    return seps < region.radius


def __cone_intersects_cone(region: ConeRegion, other: ConeRegion) -> bool:
    dtheta = region.center.separation(other.center)
    return dtheta < (region.radius + other.radius)


def __skybox_contains_point(region: SkyboxRegion, coords: SkyCoord) -> NDArray:
    ra_in_range = (coords.ra.value > region.ra_bounds[0]) & (
        coords.ra.value < region.ra_bounds[1]
    )
    dec_in_range = (coords.dec.value > region.dec_bounds[0]) & (
        coords.dec.value < region.dec_bounds[1]
    )
    return ra_in_range & dec_in_range


def __skybox_contains_skybox(region: SkyboxRegion, other: SkyboxRegion) -> bool:
    ra_contained = (
        region.ra_bounds[0] <= other.ra_bounds[0]
        and region.ra_bounds[1] >= other.ra_bounds[1]
    )
    dec_contained = (
        region.dec_bounds[0] <= other.dec_bounds[0]
        and region.dec_bounds[1] >= other.dec_bounds[1]
    )
    return ra_contained and dec_contained


def __skybox_intersects_skybox(region: SkyboxRegion, other: SkyboxRegion) -> bool:
    ra_overlaps = (
        region.ra_bounds[0] < other.ra_bounds[1]
        and region.ra_bounds[1] > other.ra_bounds[0]
    )
    dec_overlaps = (
        region.dec_bounds[0] < other.dec_bounds[1]
        and region.dec_bounds[1] > other.dec_bounds[0]
    )
    return ra_overlaps and dec_overlaps


def __skybox_contains_cone(region: SkyboxRegion, other: ConeRegion) -> bool:
    import astropy.units as u

    radius_deg = other.radius.to(u.deg).value
    ra = other.center.ra.deg
    dec = other.center.dec.deg
    ra_contained = (
        region.ra_bounds[0] <= ra - radius_deg
        and region.ra_bounds[1] >= ra + radius_deg
    )
    dec_contained = (
        region.dec_bounds[0] <= dec - radius_deg
        and region.dec_bounds[1] >= dec + radius_deg
    )
    return ra_contained and dec_contained


def __cone_contains_skybox(region: ConeRegion, other: SkyboxRegion) -> bool:
    corners = SkyCoord(
        ra=[
            other.ra_bounds[0],
            other.ra_bounds[0],
            other.ra_bounds[1],
            other.ra_bounds[1],
        ],
        dec=[
            other.dec_bounds[0],
            other.dec_bounds[1],
            other.dec_bounds[0],
            other.dec_bounds[1],
        ],
        unit="deg",
    )
    return bool(np.all(__cone_contains_point(region, corners)))


def __skybox_intersects_cone(region: SkyboxRegion, other: ConeRegion) -> bool:
    nearest_ra = np.clip(other.center.ra.deg, region.ra_bounds[0], region.ra_bounds[1])
    nearest_dec = np.clip(
        other.center.dec.deg, region.dec_bounds[0], region.dec_bounds[1]
    )
    nearest = SkyCoord(ra=nearest_ra, dec=nearest_dec, unit="deg")
    return bool(other.center.separation(nearest) < other.radius)


def __healpix_intersects_healpix(region: HealpixRegion, other: HealpixRegion) -> bool:
    return bool(np.any(np.isin(region.pixels, other.pixels)))


def __healpix_intersects_other(region: HealpixRegion, other) -> bool:
    try:
        intersections = other.get_healpix_intersections(region.nside)
        return bool(np.any(np.isin(region.pixels, intersections)))
    except AttributeError:
        raise ValueError(f"Expected a 2D Sky Region but received {type(other)}")


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def contains_3d(region, other) -> bool | NDArray:
    """
    Check whether a 3D region contains another region or a set of points.

    Points should be passed as a numpy array with shape (3, n_points).
    """
    from opencosmo.spatial.region import BoxRegion

    match (region, other):
        case (BoxRegion(), BoxRegion()):
            return __box_contains_box(region, other)
        case (BoxRegion(), arr) if isinstance(arr, np.ndarray):
            return __box_contains_points(region, arr)
        case _:
            raise ValueError(f"Expected a 3D region or point array, got {type(other)}")


def intersects_3d(region, other) -> bool:
    """
    Check whether two 3D regions intersect.
    """
    from opencosmo.spatial.region import BoxRegion

    match (region, other):
        case (BoxRegion(), BoxRegion()):
            return __box_intersects_box(region, other)
        case _:
            raise ValueError(f"Expected a 3D region, got {type(other)}")


def contains_2d(region, other):
    """
    Check whether a 2D sky region contains another region or a sky coordinate.

    A region does not contain itself — contains checks require the test object
    to be strictly interior.
    """
    from opencosmo.spatial.region import (
        ConeRegion,
        FullSkyRegion,
        HealpixRegion,
        SkyboxRegion,
    )

    match (region, other):
        case (FullSkyRegion(), FullSkyRegion()):
            return False
        case (FullSkyRegion(), _):
            return True
        case (ConeRegion(), ConeRegion()):
            return __cone_contains_cone(region, other)
        case (ConeRegion(), SkyCoord()):
            return __cone_contains_point(region, other)
        case (ConeRegion(), SkyboxRegion()):
            return __cone_contains_skybox(region, other)
        case (SkyboxRegion(), ConeRegion()):
            return __skybox_contains_cone(region, other)
        case (SkyboxRegion(), SkyboxRegion()):
            return __skybox_contains_skybox(region, other)
        case (SkyboxRegion(), SkyCoord()):
            return __skybox_contains_point(region, other)
        case (HealpixRegion(), _):
            return False
        case _:
            raise ValueError(
                f"Expected a 2D Sky Region but received {type(region)}, {type(other)}"
            )


def intersects_2d(region, other):
    """
    Check whether two 2D sky regions intersect.
    """
    from opencosmo.spatial.region import (
        ConeRegion,
        FullSkyRegion,
        HealpixRegion,
        SkyboxRegion,
    )

    match (region, other):
        case (FullSkyRegion(), FullSkyRegion()):
            return False
        case (FullSkyRegion(), _):
            return True
        case (ConeRegion(), ConeRegion()):
            return __cone_intersects_cone(region, other)
        case (ConeRegion(), SkyboxRegion()):
            return __skybox_intersects_cone(other, region)
        case (SkyboxRegion(), ConeRegion()):
            return __skybox_intersects_cone(region, other)
        case (HealpixRegion(), HealpixRegion()):
            return __healpix_intersects_healpix(region, other)
        case (HealpixRegion(), _):
            return __healpix_intersects_other(region, other)
        case (SkyboxRegion(), SkyboxRegion()):
            return __skybox_intersects_skybox(region, other)
        case _:
            raise ValueError(
                f"Expected a 2D Sky Region but received {type(region)}, {type(other)}"
            )
