from __future__ import annotations

import math

import numpy as np
import pytest
from opencosmo.spatial.query import (
    BoxQuery,
    ConeQuery,
    FullSkyQuery,
    PixelSelection,
    SkyboxQuery,
    validate_nested_promotion,
)


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
