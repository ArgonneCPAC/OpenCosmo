from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore

from opencosmo.spatial.normalize import SpatialNormalizationContext, normalize_region
from opencosmo.spatial.query import (
    BoxQuery,
    ConeQuery,
    FullSkyQuery,
    PixelSelection,
    SkyboxQuery,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

__CONTEXT_2D = SpatialNormalizationContext(2, ("ra", "dec"))
__CONTEXT_3D = SpatialNormalizationContext(3, ("x", "y", "z"))


def __normalized_box_contains(region: BoxQuery, other: BoxQuery) -> bool:
    return all(
        lower <= other_lower and upper >= other_upper
        for lower, upper, other_lower, other_upper in zip(
            region.lower, region.upper, other.lower, other.upper
        )
    )


def __normalized_box_intersects(region: BoxQuery, other: BoxQuery) -> bool:
    return all(
        lower <= other_upper and upper >= other_lower
        for lower, upper, other_lower, other_upper in zip(
            region.lower, region.upper, other.lower, other.upper
        )
    )


def __normalized_cone_radius(region: ConeQuery) -> float:
    return float(2 * np.arcsin(np.sqrt(region.max_squared_chord_distance) / 2))


def __normalized_cone_separation(region: ConeQuery, other: ConeQuery) -> float:
    return float(np.arccos(np.clip(np.dot(region.center, other.center), -1.0, 1.0)))


def __unit_vectors(ra_degrees, dec_degrees) -> NDArray:
    ra = np.radians(ra_degrees)
    dec = np.radians(dec_degrees)
    cos_dec = np.cos(dec)
    return np.asarray((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def __normalized_cone_contains_points(region: ConeQuery, points: NDArray) -> NDArray:
    center = np.asarray(region.center).reshape(3, 1)
    return (
        np.sum((points.reshape(3, -1) - center) ** 2, axis=0)
        < region.max_squared_chord_distance
    )


def __normalized_skybox_contains_points(
    region: SkyboxQuery, ra_degrees, dec_degrees
) -> NDArray:
    offset = (np.asarray(ra_degrees) - region.ra_start_degrees) % 360.0
    return (
        (offset > 0.0)
        & (offset < region.ra_width_degrees)
        & (np.asarray(dec_degrees) > region.dec_min_degrees)
        & (np.asarray(dec_degrees) < region.dec_max_degrees)
    )


def __normalized_skybox_contains(region: SkyboxQuery, other: SkyboxQuery) -> bool:
    offset = (other.ra_start_degrees - region.ra_start_degrees) % 360.0
    return (
        region.ra_width_degrees >= other.ra_width_degrees
        and offset + other.ra_width_degrees <= region.ra_width_degrees
        and region.dec_min_degrees <= other.dec_min_degrees
        and region.dec_max_degrees >= other.dec_max_degrees
    )


def __normalized_skybox_intersects(region: SkyboxQuery, other: SkyboxQuery) -> bool:
    start = (other.ra_start_degrees - region.ra_start_degrees) % 360.0
    ra_overlaps = (
        region.ra_interval == "full"
        or other.ra_interval == "full"
        or (
            start < region.ra_width_degrees
            or start - 360.0 + other.ra_width_degrees > 0.0
        )
    )
    dec_overlaps = (
        region.dec_min_degrees < other.dec_max_degrees
        and region.dec_max_degrees > other.dec_min_degrees
    )
    return ra_overlaps and dec_overlaps


def __normalized_skybox_contains_cone(region: SkyboxQuery, other: ConeQuery) -> bool:
    radius = np.degrees(__normalized_cone_radius(other))
    center = np.asarray(other.center)
    ra = float(np.degrees(np.arctan2(center[1], center[0])) % 360.0)
    dec = float(np.degrees(np.arcsin(center[2])))
    offset = (ra - radius - region.ra_start_degrees) % 360.0
    return (
        offset + 2 * radius <= region.ra_width_degrees
        and region.dec_min_degrees <= dec - radius
        and region.dec_max_degrees >= dec + radius
    )


def __normalized_cone_contains_skybox(region: ConeQuery, other: SkyboxQuery) -> bool:
    corners = __unit_vectors(
        np.array(
            [
                other.ra_start_degrees,
                other.ra_start_degrees,
                other.ra_start_degrees + other.ra_width_degrees,
                other.ra_start_degrees + other.ra_width_degrees,
            ]
        ),
        np.array(
            [
                other.dec_min_degrees,
                other.dec_max_degrees,
                other.dec_min_degrees,
                other.dec_max_degrees,
            ]
        ),
    )
    return bool(np.all(__normalized_cone_contains_points(region, corners)))


def __normalized_skybox_intersects_cone(region: SkyboxQuery, other: ConeQuery) -> bool:
    center = np.asarray(other.center)
    center_ra = float(np.degrees(np.arctan2(center[1], center[0])) % 360.0)
    center_dec = float(np.degrees(np.arcsin(center[2])))
    offset = (center_ra - region.ra_start_degrees) % 360.0
    if offset <= region.ra_width_degrees:
        nearest_ra = center_ra
    else:
        ra_end = (region.ra_start_degrees + region.ra_width_degrees) % 360.0
        start_distance = min(offset, 360.0 - offset)
        end_offset = (center_ra - ra_end) % 360.0
        end_distance = min(end_offset, 360.0 - end_offset)
        nearest_ra = (
            region.ra_start_degrees if start_distance < end_distance else ra_end
        )
    nearest_dec = np.clip(center_dec, region.dec_min_degrees, region.dec_max_degrees)
    nearest = __unit_vectors(nearest_ra, nearest_dec)
    return bool(__normalized_cone_contains_points(other, nearest)[0])


def __pixels(selection: PixelSelection) -> NDArray:
    if selection.starts.size == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate(
        [
            np.arange(start, start + size, dtype=np.int64)
            for start, size in zip(selection.starts, selection.sizes)
        ]
    )


def contains_3d(region, other) -> bool | NDArray:
    """
    Check whether a 3D region contains another region or a set of points.

    Points should be passed as a numpy array with shape (3, n_points).
    """
    from opencosmo.spatial.region import BoxRegion

    match (region, other):
        case (BoxRegion(), BoxRegion()):
            left = normalize_region(region, context=__CONTEXT_3D)
            right = normalize_region(other, context=__CONTEXT_3D)
            assert isinstance(left, BoxQuery) and isinstance(right, BoxQuery)
            return __normalized_box_contains(left, right)
        case (BoxRegion(), arr) if isinstance(arr, np.ndarray):
            query = normalize_region(region, context=__CONTEXT_3D)
            assert isinstance(query, BoxQuery)
            if arr.shape[0] != 3:
                raise ValueError(
                    "Expected a coordinate array with shape (3, n_points)!"
                )
            return np.all(
                (arr > np.asarray(query.lower)[:, None])
                & (arr < np.asarray(query.upper)[:, None]),
                axis=0,
            )
        case _:
            raise ValueError(f"Expected a 3D region or point array, got {type(other)}")


def intersects_3d(region, other) -> bool:
    """
    Check whether two 3D regions intersect.
    """
    from opencosmo.spatial.region import BoxRegion

    match (region, other):
        case (BoxRegion(), BoxRegion()):
            left = normalize_region(region, context=__CONTEXT_3D)
            right = normalize_region(other, context=__CONTEXT_3D)
            assert isinstance(left, BoxQuery) and isinstance(right, BoxQuery)
            return __normalized_box_intersects(left, right)
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
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, FullSkyQuery) and isinstance(right, FullSkyQuery)
            return False
        case (FullSkyRegion(), _):
            return True
        case (ConeRegion(), ConeRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, ConeQuery) and isinstance(right, ConeQuery)
            return __normalized_cone_radius(left) > (
                __normalized_cone_separation(left, right)
                + __normalized_cone_radius(right)
            )
        case (ConeRegion(), SkyCoord()):
            query = normalize_region(region, context=__CONTEXT_2D)
            assert isinstance(query, ConeQuery)
            return __normalized_cone_contains_points(
                query, np.asarray(other.cartesian.xyz.value)
            )
        case (ConeRegion(), SkyboxRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, ConeQuery) and isinstance(right, SkyboxQuery)
            return __normalized_cone_contains_skybox(left, right)
        case (SkyboxRegion(), ConeRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, SkyboxQuery) and isinstance(right, ConeQuery)
            return __normalized_skybox_contains_cone(left, right)
        case (SkyboxRegion(), SkyboxRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, SkyboxQuery) and isinstance(right, SkyboxQuery)
            return __normalized_skybox_contains(left, right)
        case (SkyboxRegion(), SkyCoord()):
            query = normalize_region(region, context=__CONTEXT_2D)
            assert isinstance(query, SkyboxQuery)
            return __normalized_skybox_contains_points(
                query, other.ra.deg, other.dec.deg
            )
        case (HealpixRegion(), HealpixRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, PixelSelection) and isinstance(
                right, PixelSelection
            )
            return bool(np.all(np.isin(__pixels(right), __pixels(left))))
        case (HealpixRegion(), _):
            try:
                intersections = other.get_healpix_intersections(region.nside)
            except AttributeError:
                raise ValueError(f"Expected a 2D Sky Region but received {type(other)}")
            left = normalize_region(region, context=__CONTEXT_2D)
            candidate = HealpixRegion(intersections, region.nside)
            right = normalize_region(candidate, context=__CONTEXT_2D)
            assert isinstance(left, PixelSelection) and isinstance(
                right, PixelSelection
            )
            return bool(np.all(np.isin(__pixels(right), __pixels(left))))
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
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, FullSkyQuery) and isinstance(right, FullSkyQuery)
            return False
        case (FullSkyRegion(), _):
            return True
        case (ConeRegion(), ConeRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, ConeQuery) and isinstance(right, ConeQuery)
            return __normalized_cone_separation(left, right) < (
                __normalized_cone_radius(left) + __normalized_cone_radius(right)
            )
        case (ConeRegion(), SkyboxRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, ConeQuery) and isinstance(right, SkyboxQuery)
            return __normalized_skybox_intersects_cone(right, left)
        case (SkyboxRegion(), ConeRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, SkyboxQuery) and isinstance(right, ConeQuery)
            return __normalized_skybox_intersects_cone(left, right)
        case (HealpixRegion(), HealpixRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, PixelSelection) and isinstance(
                right, PixelSelection
            )
            return bool(np.any(np.isin(__pixels(left), __pixels(right))))
        case (HealpixRegion(), _):
            try:
                intersections = other.get_healpix_intersections(region.nside)
            except AttributeError:
                raise ValueError(f"Expected a 2D Sky Region but received {type(other)}")
            left = normalize_region(region, context=__CONTEXT_2D)
            candidate = HealpixRegion(intersections, region.nside)
            right = normalize_region(candidate, context=__CONTEXT_2D)
            assert isinstance(left, PixelSelection) and isinstance(
                right, PixelSelection
            )
            return bool(np.any(np.isin(__pixels(left), __pixels(right))))
        case (SkyboxRegion(), SkyboxRegion()):
            left = normalize_region(region, context=__CONTEXT_2D)
            right = normalize_region(other, context=__CONTEXT_2D)
            assert isinstance(left, SkyboxQuery) and isinstance(right, SkyboxQuery)
            return __normalized_skybox_intersects(left, right)
        case _:
            raise ValueError(
                f"Expected a 2D Sky Region but received {type(region)}, {type(other)}"
            )
