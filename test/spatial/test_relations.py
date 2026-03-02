"""
Tests for all spatial relationship functions in opencosmo.spatial.relations.

Covers contains_3d, intersects_3d, contains_2d, and intersects_2d across
BoxRegion, ConeRegion, SkyboxRegion, HealpixRegion, and FullSkyRegion with
three relationship classes for each:
  - Contains + Intersects (one region fully inside the other)
  - Intersects but not Contains (partial overlap)
  - Neither Intersects nor Contains (completely disjoint)
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

import opencosmo as oc
from opencosmo.spatial.region import FullSkyRegion, HealpixRegion
from opencosmo.spatial.relations import (
    contains_2d,
    contains_3d,
    intersects_2d,
    intersects_3d,
)

# ---------------------------------------------------------------------------
# BoxRegion — contains_3d
# ---------------------------------------------------------------------------


class TestBoxContainsBox:
    def test_contains_and_intersects(self):
        """Small box fully inside big box: contains=True, intersects=True."""
        big = oc.make_box((0, 0, 0), (10, 10, 10))
        small = oc.make_box((2, 2, 2), (8, 8, 8))
        assert contains_3d(big, small)
        assert intersects_3d(big, small)

    def test_intersects_not_contains(self):
        """Partially overlapping boxes: contains=False, intersects=True."""
        box1 = oc.make_box((0, 0, 0), (6, 6, 6))
        box2 = oc.make_box((4, 4, 4), (10, 10, 10))
        assert not contains_3d(box1, box2)
        assert not contains_3d(box2, box1)
        assert intersects_3d(box1, box2)
        assert intersects_3d(box2, box1)

    def test_neither(self):
        """Completely disjoint boxes: contains=False, intersects=False."""
        box1 = oc.make_box((0, 0, 0), (4, 4, 4))
        box2 = oc.make_box((6, 6, 6), (10, 10, 10))
        assert not contains_3d(box1, box2)
        assert not contains_3d(box2, box1)
        assert not intersects_3d(box1, box2)
        assert not intersects_3d(box2, box1)

    def test_box_contains_itself(self):
        """_box_contains_box uses non-strict bounds so a box contains itself."""
        box = oc.make_box((0, 0, 0), (10, 10, 10))
        assert contains_3d(box, box)

    def test_containment_is_not_symmetric(self):
        """If big contains small, small does not contain big."""
        big = oc.make_box((0, 0, 0), (10, 10, 10))
        small = oc.make_box((3, 3, 3), (7, 7, 7))
        assert contains_3d(big, small)
        assert not contains_3d(small, big)


# ---------------------------------------------------------------------------
# BoxRegion — contains_3d with point arrays
# ---------------------------------------------------------------------------


class TestBoxContainsPoints:
    def setup_method(self):
        self.box = oc.make_box((0, 0, 0), (10, 10, 10))

    def test_all_points_inside(self):
        points = np.array([[2, 5, 8], [3, 5, 7], [4, 5, 6]], dtype=float)
        result = contains_3d(self.box, points)
        assert np.all(result)

    def test_mixed_points(self):
        # Two inside, two outside
        points = np.array([[2, 5, 8, 15], [3, 5, 7, 15], [4, 5, 6, 15]], dtype=float)
        result = contains_3d(self.box, points)
        assert np.array_equal(result, [True, True, True, False])

    def test_all_points_outside(self):
        points = np.array([[11, 20, -1], [11, 20, -1], [11, 20, -1]], dtype=float)
        result = contains_3d(self.box, points)
        assert not np.any(result)

    def test_wrong_shape_raises(self):
        points = np.array([[1, 2], [3, 4]], dtype=float)  # shape (2, n), not (3, n)
        with pytest.raises(ValueError):
            contains_3d(self.box, points)


# ---------------------------------------------------------------------------
# BoxRegion — error cases
# ---------------------------------------------------------------------------


class TestBoxRelationErrors:
    def test_contains_3d_invalid_type(self):
        box = oc.make_box((0, 0, 0), (10, 10, 10))
        with pytest.raises(ValueError):
            contains_3d(box, "not_a_region")

    def test_intersects_3d_invalid_type(self):
        box = oc.make_box((0, 0, 0), (10, 10, 10))
        with pytest.raises(ValueError):
            intersects_3d(box, "not_a_region")


# ---------------------------------------------------------------------------
# FullSkyRegion — contains_2d / intersects_2d
# ---------------------------------------------------------------------------


class TestFullSkyRegion:
    def setup_method(self):
        self.full = FullSkyRegion()
        self.cone = oc.make_cone((180.0, 0.0), 10.0)

    def test_full_sky_does_not_contain_itself(self):
        assert not contains_2d(self.full, FullSkyRegion())

    def test_full_sky_contains_cone(self):
        assert contains_2d(self.full, self.cone)

    def test_full_sky_does_not_intersect_itself(self):
        assert not intersects_2d(self.full, FullSkyRegion())

    def test_full_sky_intersects_cone(self):
        assert intersects_2d(self.full, self.cone)


# ---------------------------------------------------------------------------
# ConeRegion — contains_2d (cone vs cone)
# ---------------------------------------------------------------------------


class TestConeContainsCone:
    """
    Big cone: center (180, 0), radius 20 deg.
    Small cone (inside): center (185, 0), radius 5 deg.
      sep ≈ 5 deg; 20 > 5+5=10 → contains.
    Overlapping cone: center (195, 0), radius 10 deg.
      sep = 15 deg; 20 > 15+10=25 → False; 15 < 20+10=30 → intersects.
    Far cone: center (250, 0), radius 5 deg.
      sep = 70 deg; 70 >= 20+5=25 → neither.
    """

    def setup_method(self):
        self.big = oc.make_cone((180.0, 0.0), 20.0)
        self.small_inside = oc.make_cone((185.0, 0.0), 5.0)
        self.overlapping = oc.make_cone((195.0, 0.0), 10.0)
        self.far_away = oc.make_cone((250.0, 0.0), 5.0)

    def test_contains_and_intersects(self):
        assert contains_2d(self.big, self.small_inside)
        assert intersects_2d(self.big, self.small_inside)

    def test_intersects_not_contains(self):
        assert not contains_2d(self.big, self.overlapping)
        assert intersects_2d(self.big, self.overlapping)

    def test_neither(self):
        assert not contains_2d(self.big, self.far_away)
        assert not intersects_2d(self.big, self.far_away)

    def test_containment_not_symmetric(self):
        assert contains_2d(self.big, self.small_inside)
        assert not contains_2d(self.small_inside, self.big)

    def test_intersects_is_symmetric(self):
        assert intersects_2d(self.big, self.overlapping)
        assert intersects_2d(self.overlapping, self.big)


# ---------------------------------------------------------------------------
# ConeRegion — contains_2d with SkyCoord points
# ---------------------------------------------------------------------------


class TestConeContainsPoint:
    def setup_method(self):
        self.cone = oc.make_cone((180.0, 0.0), 10.0)

    def test_point_inside(self):
        point = SkyCoord(182.0, 2.0, unit="deg")
        result = contains_2d(self.cone, point)
        assert np.all(result)

    def test_point_outside(self):
        point = SkyCoord(220.0, 0.0, unit="deg")
        result = contains_2d(self.cone, point)
        assert not np.any(result)

    def test_multiple_points_mixed(self):
        points = SkyCoord([182.0, 200.0], [2.0, 0.0], unit="deg")
        result = contains_2d(self.cone, points)
        assert result[0]
        assert not result[1]


# ---------------------------------------------------------------------------
# SkyboxRegion — contains_2d with SkyCoord points
# ---------------------------------------------------------------------------


class TestSkyboxContainsPoint:
    def setup_method(self):
        # Box covers RA 170–190, Dec -10–10
        self.skybox = oc.make_skybox((170.0, -10.0), (190.0, 10.0))

    def test_point_inside(self):
        point = SkyCoord(180.0, 0.0, unit="deg")
        result = contains_2d(self.skybox, point)
        assert np.all(result)

    def test_point_outside_ra(self):
        point = SkyCoord(200.0, 0.0, unit="deg")
        result = contains_2d(self.skybox, point)
        assert not np.any(result)

    def test_point_outside_dec(self):
        point = SkyCoord(180.0, 20.0, unit="deg")
        result = contains_2d(self.skybox, point)
        assert not np.any(result)

    def test_multiple_points_mixed(self):
        points = SkyCoord([180.0, 200.0, 180.0], [0.0, 0.0, 30.0], unit="deg")
        result = contains_2d(self.skybox, points)
        assert result[0]
        assert not result[1]
        assert not result[2]

    def test_skybox_intersects_unsupported_type_raises(self):
        cone = oc.make_cone((180.0, 0.0), 5.0)
        with pytest.raises(ValueError):
            intersects_2d(self.skybox, cone)


# ---------------------------------------------------------------------------
# SkyboxRegion — contains_2d / intersects_2d (skybox vs skybox)
# ---------------------------------------------------------------------------


class TestSkyboxContainsSkybox:
    """
    Big skybox: RA 170–190, Dec -10–10.
    Small inside: RA 175–185, Dec -5–5  → big contains small.
    Overlapping: RA 180–200, Dec -5–5   → intersects but not contains.
    Disjoint: RA 200–220, Dec -10–10    → neither.
    """

    def setup_method(self):
        self.big = oc.make_skybox((170.0, -10.0), (190.0, 10.0))
        self.small_inside = oc.make_skybox((175.0, -5.0), (185.0, 5.0))
        self.overlapping = oc.make_skybox((180.0, -5.0), (200.0, 5.0))
        self.disjoint = oc.make_skybox((200.0, -10.0), (220.0, 10.0))

    def test_contains_and_intersects(self):
        assert contains_2d(self.big, self.small_inside)
        assert intersects_2d(self.big, self.small_inside)

    def test_intersects_not_contains(self):
        assert not contains_2d(self.big, self.overlapping)
        assert intersects_2d(self.big, self.overlapping)

    def test_neither(self):
        assert not contains_2d(self.big, self.disjoint)
        assert not intersects_2d(self.big, self.disjoint)

    def test_containment_not_symmetric(self):
        assert contains_2d(self.big, self.small_inside)
        assert not contains_2d(self.small_inside, self.big)

    def test_intersects_is_symmetric(self):
        assert intersects_2d(self.big, self.overlapping)
        assert intersects_2d(self.overlapping, self.big)


# ---------------------------------------------------------------------------
# HealpixRegion — contains_2d / intersects_2d
# ---------------------------------------------------------------------------


class TestHealpixRegion:
    NSIDE = 16

    def setup_method(self):
        pixels_a = np.arange(0, 50, dtype=np.int_)
        pixels_b = np.arange(25, 75, dtype=np.int_)  # overlaps [25–49] with a
        pixels_c = np.arange(100, 150, dtype=np.int_)  # completely disjoint from a

        self.region_a = HealpixRegion(pixels_a, self.NSIDE)
        self.region_b = HealpixRegion(pixels_b, self.NSIDE)
        self.region_c = HealpixRegion(pixels_c, self.NSIDE)

    # HealpixRegion never reports containment
    def test_healpix_never_contains_healpix(self):
        assert not contains_2d(self.region_a, self.region_b)

    def test_healpix_never_contains_cone(self):
        cone = oc.make_cone((180.0, 0.0), 10.0)
        assert not contains_2d(self.region_a, cone)

    # intersects_2d: HealpixRegion vs HealpixRegion
    def test_healpix_intersects_healpix_overlapping(self):
        assert intersects_2d(self.region_a, self.region_b)
        assert intersects_2d(self.region_b, self.region_a)

    def test_healpix_not_intersects_healpix_disjoint(self):
        assert not intersects_2d(self.region_a, self.region_c)
        assert not intersects_2d(self.region_c, self.region_a)

    # intersects_2d: HealpixRegion vs ConeRegion (via _healpix_intersects_other)
    def test_healpix_intersects_cone_overlapping(self):
        # Build a healpix region from pixels that the cone actually covers
        cone = oc.make_cone((180.0, 0.0), 5.0)
        cone_pixels = cone.get_healpix_intersections(self.NSIDE)
        region = HealpixRegion(cone_pixels, self.NSIDE)
        assert intersects_2d(region, cone)

    def test_healpix_not_intersects_cone_disjoint(self):
        # Cone near the north pole; our region is at pixels 0-49 which are near
        # the equator for nside=16 — use a small cone far from those pixels
        cone = oc.make_cone((0.0, 89.0), 0.5)  # tiny cone near north pole
        # Pixels 0-49 at nside=16 are nowhere near the north pole
        assert not intersects_2d(self.region_a, cone)

    def test_healpix_intersects_invalid_type_raises(self):
        with pytest.raises(ValueError):
            intersects_2d(self.region_a, "not_a_region")
