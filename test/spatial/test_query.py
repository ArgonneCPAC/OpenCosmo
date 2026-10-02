from __future__ import annotations

import math

import pytest
from opencosmo.spatial.query import BoxQuery, ConeQuery, FullSkyQuery, SkyboxQuery


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
