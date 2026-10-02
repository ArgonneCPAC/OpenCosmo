from __future__ import annotations

import math

import numpy as np
import pytest
from opencosmo.spatial.normalize import SpatialNormalizationContext, normalize_region
from opencosmo.spatial.query import (
    BoxQuery,
    ConeQuery,
    FullSkyQuery,
    PixelSelection,
    SkyboxQuery,
    validate_nested_promotion,
)
from opencosmo.spatial.region import BoxRegion, ConeRegion, FullSkyRegion, HealpixRegion


def test_box_query_accepts_finite_ordered_bounds() -> None:
    query = BoxQuery((0, 1, 2), (3, 4, 5))

    assert query.lower == (0.0, 1.0, 2.0)
    assert query.upper == (3.0, 4.0, 5.0)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ((0, 1), (3, 4, 5)),
        ((0, 1, 2), (3, 4, math.inf)),
        ((0, 1, 2), (0, 4, 5)),
    ],
)
def test_box_query_rejects_invalid_bounds(
    lower: tuple[float, ...], upper: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        BoxQuery(lower, upper)  # type: ignore[arg-type]


def test_cone_query_accepts_finite_threshold() -> None:
    query = ConeQuery((1, 0, 0), 0)

    assert query.center == (1.0, 0.0, 0.0)
    assert query.max_squared_chord_distance == 0.0


@pytest.mark.parametrize("threshold", [-1, math.inf, math.nan])
def test_cone_query_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="max_squared_chord_distance"):
        ConeQuery((1, 0, 0), threshold)


def test_skybox_query_validates_declination_bounds() -> None:
    query = SkyboxQuery(350, 20, -10, 10)

    assert query.ra_start_degrees == 350
    assert query.ra_width_degrees == 20

    with pytest.raises(ValueError, match="declination"):
        SkyboxQuery(0, 10, -91, 10)


def test_full_sky_query_is_immutable() -> None:
    query = FullSkyQuery()

    with pytest.raises(AttributeError):
        object.__setattr__(query, "value", 1)


def test_pixel_selection_copies_and_freezes_canonical_ranges() -> None:
    starts = np.array([1, 5], dtype=np.int64)
    sizes = np.array([2, 3], dtype=np.int64)
    selection = PixelSelection(2, "nested", starts, sizes)

    starts[0] = 10
    sizes[0] = 10

    assert np.array_equal(selection.starts, [1, 5])
    assert np.array_equal(selection.sizes, [2, 3])
    with pytest.raises(ValueError):
        selection.starts[0] = 10


@pytest.mark.parametrize(
    ("starts", "sizes"),
    [
        (np.array([1], dtype=np.int32), np.array([1], dtype=np.int32)),
        (np.array([[1]], dtype=np.int64), np.array([1], dtype=np.int64)),
        (np.array([1], dtype=np.int64), np.array([1, 2], dtype=np.int64)),
        (np.array([-1], dtype=np.int64), np.array([1], dtype=np.int64)),
        (np.array([1], dtype=np.int64), np.array([0], dtype=np.int64)),
        (np.array([2, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)),
        (np.array([1, 2], dtype=np.int64), np.array([2, 1], dtype=np.int64)),
    ],
)
def test_pixel_selection_rejects_noncanonical_ranges(
    starts: np.ndarray, sizes: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        PixelSelection(1, "nested", starts, sizes)


def test_pixel_selection_rejects_invalid_metadata_and_bounds() -> None:
    empty = np.array([], dtype=np.int64)

    with pytest.raises(ValueError, match="nside"):
        PixelSelection(3, "nested", empty, empty)
    with pytest.raises(ValueError, match="ordering"):
        PixelSelection(1, "invalid", empty, empty)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exceed"):
        PixelSelection(
            1,
            "nested",
            np.array([11], dtype=np.int64),
            np.array([2], dtype=np.int64),
        )


def test_pixel_selection_accepts_empty_ranges() -> None:
    empty = np.array([], dtype=np.int64)

    selection = PixelSelection(1, "ring", empty, empty)

    assert selection.starts.size == 0
    assert selection.sizes.size == 0


def test_validate_nested_promotion() -> None:
    assert validate_nested_promotion(2, 8) == 4

    with pytest.raises(ValueError, match="target"):
        validate_nested_promotion(8, 2)


def test_normalization_context_copies_coordinate_names() -> None:
    names = ["x", "y", "z"]
    context = SpatialNormalizationContext(3, names)  # type: ignore[arg-type]

    names[0] = "other"

    assert context.coordinate_names == ("x", "y", "z")


@pytest.mark.parametrize(
    ("dimensions", "coordinate_names"),
    [(1, ("x",)), (2, ("x",)), (3, ("x", "y"))],
)
def test_normalization_context_validates_dimensions_and_names(
    dimensions: int, coordinate_names: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        SpatialNormalizationContext(dimensions, coordinate_names)  # type: ignore[arg-type]


def test_normalize_box_region() -> None:
    context = SpatialNormalizationContext(3, ("x", "y", "z"))

    query = normalize_region(BoxRegion((1, 2, 3), (1, 2, 3)), context=context)

    assert query == BoxQuery((0, 0, 0), (2, 4, 6))


def test_normalize_healpix_region_canonicalizes_pixel_ranges() -> None:
    context = SpatialNormalizationContext(2, ("ra", "dec"))
    region = HealpixRegion(np.array([1, 2, 4, 5, 6], dtype=np.int64), 1)

    query = normalize_region(region, context=context)

    assert isinstance(query, PixelSelection)
    assert np.array_equal(query.starts, [1, 4])
    assert np.array_equal(query.sizes, [2, 3])


def test_normalize_full_sky_region() -> None:
    context = SpatialNormalizationContext(2, ("ra", "dec"))

    assert normalize_region(FullSkyRegion(), context=context) == FullSkyQuery()


def test_normalize_region_rejects_incompatible_context() -> None:
    context = SpatialNormalizationContext(2, ("ra", "dec"))

    with pytest.raises(ValueError, match="three-dimensional"):
        normalize_region(BoxRegion((1, 2, 3), (1, 1, 1)), context=context)


def test_normalize_cone_region_uses_unit_vector_and_squared_chord_distance() -> None:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    context = SpatialNormalizationContext(2, ("ra", "dec"))
    region = ConeRegion(SkyCoord(0 * u.deg, 0 * u.deg), 60 * u.deg)

    query = normalize_region(region, context=context)

    assert isinstance(query, ConeQuery)
    assert np.allclose(query.center, (1, 0, 0))
    assert query.max_squared_chord_distance == pytest.approx(1)
