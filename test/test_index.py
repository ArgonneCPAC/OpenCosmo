import numpy as np
import pytest
from opencosmo.index import coalesce_chunks, combine, into_array


def test_combine_chunked_sorts_and_coalesces_disjoint_ranges():
    a = (np.array([8, 0]), np.array([2, 2]))
    b = (np.array([2, 6]), np.array([2, 2]))
    starts, sizes = combine(a, b)
    np.testing.assert_array_equal(starts, [0, 6])
    np.testing.assert_array_equal(sizes, [4, 4])


def test_combine_empty_chunked_indices():
    empty = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    starts, sizes = combine(empty, empty)
    assert starts.dtype == sizes.dtype == np.int64
    assert len(starts) == len(sizes) == 0


def test_combine_simple_and_chunked_deduplicates():
    chunks = (np.array([2, 8]), np.array([3, 2]))
    simple = np.array([9, 3, 1, 1])
    for a, b in ((chunks, simple), (simple, chunks)):
        result = combine(a, b)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 8, 9])


def test_combine_simple_indices_deduplicates():
    result = combine(np.array([4, 1, 1]), np.array([4, 3]))
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 3, 4])


def __chunks(*pairs):
    return (
        np.array([p[0] for p in pairs], dtype=np.int64),
        np.array([p[1] for p in pairs], dtype=np.int64),
    )


def __as_pairs(index):
    return list(zip(index[0].tolist(), index[1].tolist()))


@pytest.mark.parametrize(
    "a,b,expected",
    [
        # overlapping
        ([(0, 5)], [(2, 5)], [(0, 7)]),
        # nested
        ([(0, 10)], [(2, 3)], [(0, 10)]),
        ([(2, 3)], [(0, 10)], [(0, 10)]),
        # duplicates
        ([(1, 2), (1, 2)], [(1, 2)], [(1, 2)]),
        # adjacent
        ([(0, 2)], [(2, 2)], [(0, 4)]),
        # unsorted, multiple groups
        ([(20, 2), (0, 3)], [(10, 1), (2, 2), (11, 1)], [(0, 4), (10, 2), (20, 2)]),
        # empty ranges are dropped and do not bridge gaps
        ([(0, 0), (3, 2)], [(3, 0), (5, 0), (10, 0)], [(3, 2)]),
        ([(5, 0)], [(5, 0)], []),
        # one side empty
        ([], [(4, 2), (0, 1)], [(0, 1), (4, 2)]),
        # one chunk bridging several
        ([(0, 1), (3, 1), (6, 1)], [(0, 7)], [(0, 7)]),
    ],
)
def test_combine_chunked_union(a, b, expected):
    result = combine(__chunks(*a), __chunks(*b))
    assert isinstance(result, tuple)
    assert result[0].dtype == result[1].dtype == np.int64
    assert __as_pairs(result) == expected


def test_combine_overlapping_chunks_is_a_union():
    a = (np.array([0]), np.array([5]))
    b = (np.array([2]), np.array([5]))
    result = combine(a, b)
    assert isinstance(result, tuple)
    np.testing.assert_array_equal(into_array(result), np.arange(7))


def test_combine_chunked_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = __chunks(*zip(rng.integers(0, 60, 8), rng.integers(0, 8, 8)))
        b = __chunks(*zip(rng.integers(0, 60, 8), rng.integers(0, 8, 8)))
        result = combine(a, b)
        expected = np.union1d(into_array(a), into_array(b))
        np.testing.assert_array_equal(into_array(result), expected)
        ends = result[0] + result[1]
        assert np.all(result[1] > 0)
        assert np.all(result[0][1:] > ends[:-1])


def test_combine_chunked_negative_starts():
    result = combine(__chunks((-5, 2)), __chunks((-3, 3), (2, 1)))
    assert __as_pairs(result) == [(-5, 5), (2, 1)]


def test_combine_chunked_strided_arrays():
    starts = np.array([4, 99, 0, 99], dtype=np.int64)[::2]
    sizes = np.array([2, 99, 2, 99], dtype=np.int64)[::2]
    assert __as_pairs(combine((starts, sizes), __chunks((2, 2)))) == [(0, 6)]


def test_combine_chunked_rejects_negative_sizes():
    with pytest.raises(ValueError, match="non-negative"):
        combine(__chunks((0, -1)), __chunks((1, 1)))


def test_combine_chunked_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        combine((np.array([0, 1]), np.array([1])), __chunks((1, 1)))


def test_combine_chunked_does_not_mutate_inputs():
    a = __chunks((5, 2), (0, 2))
    b = __chunks((1, 3))
    a_copy = tuple(x.copy() for x in a)
    combine(a, b)
    for x, y in zip(a, a_copy):
        np.testing.assert_array_equal(x, y)


def test_coalesce_merges_adjacent():
    starts, sizes = coalesce_chunks(np.array([0, 1, 2]), np.array([1, 1, 1]))
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(sizes, [3])


def test_coalesce_keeps_non_adjacent_split():
    starts, sizes = coalesce_chunks(np.array([0, 5]), np.array([1, 1]))
    np.testing.assert_array_equal(starts, [0, 5])
    np.testing.assert_array_equal(sizes, [1, 1])


def test_coalesce_mixed_runs():
    # [0,1,2] contiguous -> one chunk; gap; [10,11] contiguous -> one chunk
    starts, sizes = coalesce_chunks(
        np.array([0, 1, 2, 10, 11]), np.array([1, 1, 1, 1, 1])
    )
    np.testing.assert_array_equal(starts, [0, 10])
    np.testing.assert_array_equal(sizes, [3, 2])


def test_coalesce_varied_sizes():
    # 0->3->8->10 is fully contiguous, so all three fuse into one chunk.
    starts, sizes = coalesce_chunks(np.array([0, 3, 8]), np.array([3, 5, 2]))
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(sizes, [10])

    # A gap after the first run keeps it separate.
    starts, sizes = coalesce_chunks(np.array([0, 3, 20]), np.array([3, 5, 2]))
    np.testing.assert_array_equal(starts, [0, 20])
    np.testing.assert_array_equal(sizes, [8, 2])


def test_coalesce_empty():
    starts, sizes = coalesce_chunks(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    )
    assert len(starts) == 0
    assert len(sizes) == 0


def test_coalesce_single():
    starts, sizes = coalesce_chunks(np.array([7]), np.array([4]))
    np.testing.assert_array_equal(starts, [7])
    np.testing.assert_array_equal(sizes, [4])


def test_coalesce_preserves_array_order():
    # Locally contiguous but not globally ascending: merging must not reorder rows.
    starts, sizes = coalesce_chunks(np.array([10, 11, 0, 1]), np.array([1, 1, 1, 1]))
    np.testing.assert_array_equal(starts, [10, 0])
    np.testing.assert_array_equal(sizes, [2, 2])


def test_coalesce_output_dtype_is_int64():
    starts, sizes = coalesce_chunks(
        np.array([0, 1], dtype=np.int32), np.array([1, 1], dtype=np.int32)
    )
    assert starts.dtype == np.int64
    assert sizes.dtype == np.int64
