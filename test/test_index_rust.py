import numpy as np
import pytest
from opencosmo._lib import index as idxlib


def index(values):
    return np.array(values, dtype=np.int64)


def test_get_index_ranges():
    assert idxlib.get_simple_range(index([4, 1, 9])) == (1, 9)
    assert idxlib.get_simple_range(index([])) == (0, 0)
    assert idxlib.get_chunked_range(index([2, 10]), index([3, 4])) == (2, 14)
    assert idxlib.get_chunked_range(index([]), index([])) == (0, 0)


def test_count_chunked_ranges_uses_half_open_intervals():
    result = idxlib.n_in_range_chunked(
        index([0, 10]),
        index([5, 5]),
        index([2, 5, 8]),
        index([4, 5, 4]),
    )

    np.testing.assert_array_equal(result, [3, 0, 2])


def test_expand_chunked_index():
    result = idxlib.chunked_into_array(index([2, 10]), index([3, 2]))

    np.testing.assert_array_equal(result, [2, 3, 4, 10, 11])
    assert idxlib.chunked_into_array(index([]), index([])).dtype == np.int64
    np.testing.assert_array_equal(
        idxlib.chunked_into_array(index([2, 10]), index([0, 2])), [10, 11]
    )


def test_take_chunked_from_simple():
    result = idxlib.take_chunked_from_simple(
        index([10, 11, 12, 13, 14]), index([1, 4]), index([2, 1])
    )

    np.testing.assert_array_equal(result, [11, 12, 14])
    assert idxlib.take_chunked_from_simple(index([10]), index([]), index([])).size == 0


def test_take_chunked_from_chunked():
    starts, sizes = idxlib.take_chunked_from_chunked(
        index([0, 10]), index([5, 5]), index([3, 8]), index([4, 2])
    )

    np.testing.assert_array_equal(starts, [3, 10, 13])
    np.testing.assert_array_equal(sizes, [2, 2, 2])
    empty_starts, empty_sizes = idxlib.take_chunked_from_chunked(
        index([0]), index([5]), index([]), index([])
    )
    assert empty_starts.size == empty_sizes.size == 0


def test_reindex_column_uses_last_duplicate():
    result = idxlib.reindex_column(index([10, 20, 10]), index([10, 99, 20]))

    np.testing.assert_array_equal(result, [2, -1, 1])


def test_rebuild_chunked_by_ranges():
    result = idxlib.rebuild_chunked_by_ranges(
        index([0, 10]), index([5, 5]), index([2, 8]), index([4, 5])
    )

    assert len(result) == 2
    np.testing.assert_array_equal(result[0][0], [0])
    np.testing.assert_array_equal(result[0][1], [3])
    np.testing.assert_array_equal(result[1][0], [2])
    np.testing.assert_array_equal(result[1][1], [3])


def test_rebuild_simple_by_ranges():
    result = idxlib.rebuild_simple_by_ranges(
        index([1, 3, 10, 12]), index([0, 10]), index([5, 4])
    )

    np.testing.assert_array_equal(result[0], [1, 3])
    np.testing.assert_array_equal(result[1], [0, 2])


def test_project_chunked_on_simple_uses_half_open_intervals():
    result = idxlib.project_chunked_on_simple(
        index([0, 2, 3, 11]), index([0, 10]), index([3, 2])
    )

    np.testing.assert_array_equal(result, [0, 1, 3])


@pytest.mark.parametrize(
    "operation",
    [
        lambda: idxlib.get_chunked_range(index([0]), index([])),
        lambda: idxlib.n_in_range_chunked(
            index([0]), index([1]), index([0]), index([])
        ),
        lambda: idxlib.chunked_into_array(index([0]), index([])),
        lambda: idxlib.take_chunked_from_simple(index([0]), index([0]), index([])),
        lambda: idxlib.take_chunked_from_chunked(
            index([0]), index([1]), index([0]), index([])
        ),
        lambda: idxlib.rebuild_chunked_by_ranges(
            index([0]), index([]), index([0]), index([1])
        ),
        lambda: idxlib.rebuild_simple_by_ranges(index([0]), index([0]), index([])),
        lambda: idxlib.project_chunked_on_simple(index([0]), index([0]), index([])),
    ],
)
def test_chunked_operations_reject_mismatched_arrays(operation):
    with pytest.raises(ValueError, match="same length"):
        operation()


def test_take_chunked_from_simple_rejects_out_of_bounds_range():
    with pytest.raises(ValueError, match="outside of the range"):
        idxlib.take_chunked_from_simple(index([10, 11]), index([1]), index([2]))


def test_index_bindings_validate_shared_array_contract():
    with pytest.raises(TypeError, match="numpy array"):
        idxlib.get_simple_range([1, 2])
    with pytest.raises(TypeError, match="Invalid element type"):
        idxlib.get_simple_range(np.array([1, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="1 dimensional"):
        idxlib.get_simple_range(np.array([[1, 2]], dtype=np.int64))
