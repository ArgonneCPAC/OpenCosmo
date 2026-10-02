import numpy as np
import pytest
from opencosmo._lib import spatial as spatlib


def test_closest_distance_uses_squared_distance_threshold():
    check = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    query = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    result = spatlib.get_closest_squared_distance_3d(check, query, 9.0)

    np.testing.assert_allclose(result, [0.0, 9.0, 4.0])
    assert np.isnan(spatlib.get_closest_squared_distance_3d(check, query[1:], 8.9)[0])


def test_closest_distance_handles_empty_inputs():
    point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    empty = np.empty((0, 3), dtype=np.float64)

    assert spatlib.get_closest_squared_distance_3d(point, empty, 1.0).shape == (0,)
    assert np.isnan(spatlib.get_closest_squared_distance_3d(empty, point, 1.0)).all()


def test_closest_distance_accepts_duplicate_points():
    check = np.zeros((2, 3), dtype=np.float64)
    query = np.zeros((1, 3), dtype=np.float64)

    np.testing.assert_array_equal(
        spatlib.get_closest_squared_distance_3d(check, query, 0.0), [0.0]
    )


@pytest.mark.parametrize("threshold", [-1.0, np.nan, np.inf])
def test_closest_distance_rejects_invalid_threshold(threshold):
    points = np.zeros((1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        spatlib.get_closest_squared_distance_3d(points, points, threshold)


@pytest.mark.parametrize("argument", ["check", "query"])
def test_closest_distance_rejects_nonfinite_points(argument):
    check = np.zeros((1, 3), dtype=np.float64)
    query = np.zeros((1, 3), dtype=np.float64)
    if argument == "check":
        check[0, 0] = np.nan
    else:
        query[0, 0] = np.inf

    with pytest.raises(ValueError, match=f"{argument}_vecs.*finite"):
        spatlib.get_closest_squared_distance_3d(check, query, 1.0)


def test_closest_distance_validates_array_shape_and_dtype():
    points = np.zeros((1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="3d points"):
        spatlib.get_closest_squared_distance_3d(
            np.zeros((1, 2), dtype=np.float64), points, 1.0
        )
    with pytest.raises(TypeError, match="Invalid element type"):
        spatlib.get_closest_squared_distance_3d(
            np.zeros((1, 3), dtype=np.float32), points, 1.0
        )


def test_closest_distance_compatibility_binding():
    check = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    np.testing.assert_array_equal(
        spatlib.get_closest_distance_3d(check, query, 1.0), [1.0]
    )
