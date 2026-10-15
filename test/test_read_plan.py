import numpy as np
import pytest
from opencosmo.index import into_array
from opencosmo.index.read_plan import plan, scatter


def read(data, index, chunk_rows, budget):
    """Replay a read plan against an in-memory column."""
    row_bytes = data.dtype.itemsize * int(np.prod(data.shape[1:]))
    read_plan = plan(index, len(data), chunk_rows, row_bytes, budget)
    out = np.empty((read_plan.length,) + data.shape[1:], data.dtype)
    windows = read_plan.windows
    for w in range(len(windows.row_lo)):
        scatter(read_plan, w, data[windows.row_lo[w] : windows.row_hi[w]], out)
    return out if read_plan.gather is None else out[read_plan.gather]


def chunked(starts, sizes):
    return (np.array(starts, dtype=np.int64), np.array(sizes, dtype=np.int64))


def simple(rows):
    return np.array(rows, dtype=np.int64)


def random_chunked(rng, n_rows, n_segments, mean_size):
    starts = np.sort(rng.choice(n_rows, n_segments, replace=False))
    sizes = rng.integers(1, 2 * mean_size, n_segments)
    sizes = np.minimum(starts + sizes, np.append(starts[1:], n_rows)) - starts
    return chunked(starts, sizes)


def window_chunks(read_plan, chunk_rows):
    windows = read_plan.windows
    return [
        np.arange(lo // chunk_rows, -(-hi // chunk_rows))
        for lo, hi in zip(windows.row_lo, windows.row_hi)
    ]


DATA = np.arange(1000, dtype=np.float32)
WIDE = np.arange(1000 * 7, dtype=np.int32).reshape(1000, 7)

INDICES = {
    "empty_simple": simple([]),
    "empty_chunked": chunked([], []),
    "single_row": simple([417]),
    "first_and_last_row": simple([0, 999]),
    "sorted_simple": simple([3, 4, 5, 90, 91, 500, 998, 999]),
    "unsorted_simple": simple([500, 3, 4, 999, 91, 90]),
    "repeated_simple": simple([7, 7, 8, 3, 8, 640]),
    "chunked": chunked([10, 95, 300], [5, 20, 400]),
    "adjacent_chunked": chunked([10, 15, 20], [5, 5, 5]),
    "permuted_chunked": chunked([300, 10, 95], [400, 5, 20]),
    "zero_size_chunks": chunked([10, 20, 20, 30], [0, 5, 0, 0]),
    "overlapping_chunked": chunked([10, 12], [5, 5]),
    "full": chunked([0], [1000]),
}


@pytest.mark.parametrize("name", sorted(INDICES))
@pytest.mark.parametrize("chunk_rows", [1, 64, 1000, 4096])
@pytest.mark.parametrize("budget", [1, 1024, 1 << 30])
def test_plan_matches_fancy_indexing(name, chunk_rows, budget):
    index = INDICES[name]
    for data in (DATA, WIDE):
        result = read(data, index, chunk_rows, budget)
        np.testing.assert_array_equal(result, data[into_array(index)])


@pytest.mark.parametrize("seed", range(5))
def test_random_plan_matches_fancy_indexing(seed):
    rng = np.random.default_rng(seed)
    data = np.arange(100_000, dtype=np.int64)
    budget = 3 * 4096 * 8
    index = random_chunked(rng, len(data), 300, 50)
    np.testing.assert_array_equal(
        read(data, index, 4096, budget), data[into_array(index)]
    )

    rows = np.sort(rng.choice(len(data), 2000, replace=False))
    np.testing.assert_array_equal(read(data, rows, 4096, budget), data[rows])

    rows = rng.choice(len(data), 2000)
    np.testing.assert_array_equal(read(data, rows, 4096, budget), data[rows])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kind", ["simple", "chunked"])
def test_windows_read_each_touched_chunk_once(seed, kind):
    rng = np.random.default_rng(seed)
    n_rows, chunk_rows = 100_000, 1000
    index = random_chunked(rng, n_rows, 200, 30)
    if kind == "simple":
        index = into_array(index)
    read_plan = plan(index, n_rows, chunk_rows, 4, 4 * chunk_rows * 3)

    chunks = window_chunks(read_plan, chunk_rows)
    touched = np.unique(into_array(index) // chunk_rows)
    np.testing.assert_array_equal(np.concatenate(chunks), touched)
    assert max(map(len, chunks)) <= 3


def test_sorted_index_is_planned_without_copying():
    index = simple([3, 4, 90])
    read_plan = plan(index, 100, 10, 4, 1 << 20)
    assert read_plan.index is index
    assert read_plan.gather is None


def test_plan_size_follows_windows_not_index():
    index = np.arange(1_000_000, dtype=np.int64)
    read_plan = plan(index, len(index), 1_000_000, 4, 1 << 30)
    windows = read_plan.windows
    np.testing.assert_array_equal(windows.row_lo, [0])
    np.testing.assert_array_equal(windows.row_hi, [1_000_000])
    np.testing.assert_array_equal(windows.pos_lo, [0])
    np.testing.assert_array_equal(windows.pos_hi, [1_000_000])
    np.testing.assert_array_equal(windows.out_lo, [0])
    np.testing.assert_array_equal(windows.out_hi, [1_000_000])


def test_window_is_at_least_one_chunk():
    read_plan = plan(chunked([0], [100]), 100, 10, 8, 1)
    assert [len(c) for c in window_chunks(read_plan, 10)] == [1] * 10


def test_window_spans_only_selected_rows():
    read_plan = plan(simple([93, 95]), 98, 10, 4, 1 << 20)
    np.testing.assert_array_equal(read_plan.windows.row_lo, [93])
    np.testing.assert_array_equal(read_plan.windows.row_hi, [96])
    np.testing.assert_array_equal(read_plan.windows.out_lo, [0])
    np.testing.assert_array_equal(read_plan.windows.out_hi, [2])


@pytest.mark.parametrize(
    "index,dense",
    [
        (simple([4, 5, 6, 30, 32]), [True, False]),
        (chunked([4, 30], [3, 3]), [True, True]),
        (chunked([4, 8], [3, 3]), [False]),
    ],
)
def test_output_span_marks_fully_selected_windows(index, dense):
    windows = plan(index, 100, 10, 4, 1 << 20).windows
    row_span = windows.row_hi - windows.row_lo
    np.testing.assert_array_equal(windows.out_hi - windows.out_lo == row_span, dense)


def test_adjacent_touched_chunks_share_a_window():
    # Rows 9 and 10 touch chunks 0 and 1; row 35 touches chunk 3 after a gap.
    read_plan = plan(simple([9, 10, 35]), 100, 10, 4, 1 << 20)
    np.testing.assert_array_equal(read_plan.windows.row_lo, [9, 35])
    np.testing.assert_array_equal(read_plan.windows.row_hi, [11, 36])
    np.testing.assert_array_equal(read_plan.windows.pos_lo, [0, 2])
    np.testing.assert_array_equal(read_plan.windows.pos_hi, [2, 3])


def test_segment_spanning_windows_belongs_to_each():
    # One budget-limited window per chunk; the segment covers chunks 0-2.
    read_plan = plan(chunked([5, 40], [20, 2]), 100, 10, 4, 40)
    windows = read_plan.windows
    np.testing.assert_array_equal(windows.row_lo, [5, 10, 20, 40])
    np.testing.assert_array_equal(windows.row_hi, [10, 20, 25, 42])
    np.testing.assert_array_equal(windows.pos_lo, [0, 0, 0, 1])
    np.testing.assert_array_equal(windows.pos_hi, [1, 1, 1, 2])
    np.testing.assert_array_equal(windows.out_lo, [0, 5, 15, 20])
    np.testing.assert_array_equal(windows.out_hi, [5, 15, 20, 22])


@pytest.mark.parametrize(
    "index", [simple([5, 2]), simple([2, 2]), chunked([10, 0], [2, 2])]
)
def test_out_of_order_index_uses_gather(index):
    read_plan = plan(index, 100, 10, 4, 1 << 20)
    rows = into_array(index)
    np.testing.assert_array_equal(read_plan.index, np.unique(rows))
    np.testing.assert_array_equal(read_plan.index[read_plan.gather], rows)


def test_plan_rejects_invalid_index_type():
    with pytest.raises(TypeError, match="Invalid index type"):
        plan([1, 2, 3], 100, 10, 4, 1 << 20)


@pytest.mark.parametrize(
    "index",
    [simple([-1]), simple([100]), simple([5, 100]), simple([7, 3, 100])]
    + [chunked([95], [6]), chunked([-1], [2])],
)
def test_plan_rejects_out_of_range_index(index):
    with pytest.raises(ValueError, match="out of range"):
        plan(index, 100, 10, 4, 1 << 20)


def test_plan_rejects_negative_chunk_size():
    with pytest.raises(ValueError, match="non-negative"):
        plan(chunked([5], [-1]), 100, 10, 4, 1 << 20)


@pytest.mark.parametrize(
    "chunk_rows,row_bytes,budget", [(0, 4, 1), (10, 0, 1), (10, 4, 0)]
)
def test_plan_rejects_non_positive_parameters(chunk_rows, row_bytes, budget):
    with pytest.raises(ValueError, match="must be positive"):
        plan(simple([1]), 100, chunk_rows, row_bytes, budget)
