from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import opencosmo.handler.hdf5 as handler_module
import opencosmo.index.plan as plan_module
import opencosmo.io.writer as writer_module
import pytest
from opencosmo._lib import plan as planlib
from opencosmo.handler.hdf5 import Hdf5Handler
from opencosmo.index import build_plans, get_data, into_array, read_with_plan
from opencosmo.io.writer import Hdf5Source

if TYPE_CHECKING:
    from pathlib import Path

N_ROWS = 10_000
HDF5_CHUNK = 64


@pytest.fixture(params=[(HDF5_CHUNK,), None], ids=["chunked", "contiguous"])
def columns(request, tmp_path: Path):
    """A 1-D float64 column and a 2-D int32 column, chunked or contiguous."""
    path = tmp_path / "columns.hdf5"
    chunks = request.param
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "flat", data=np.arange(N_ROWS, dtype=np.float64), chunks=chunks
        )
        f.create_dataset(
            "wide",
            data=np.arange(3 * N_ROWS, dtype=np.int32).reshape(-1, 3),
            chunks=None if chunks is None else (HDF5_CHUNK, 3),
        )
    with h5py.File(path, "r") as f:
        yield f


@pytest.fixture
def read_sizes(monkeypatch) -> list[int]:
    """Record the number of rows in every HDF5 read issued by the index layer."""
    sizes: list[int] = []
    original = h5py.Dataset.read_direct

    def recording_read_direct(
        self: h5py.Dataset, dest: np.ndarray, source_sel: slice, dest_sel: slice
    ) -> None:
        sizes.append(source_sel.stop - source_sel.start)
        return original(self, dest, source_sel, dest_sel)

    monkeypatch.setattr(h5py.Dataset, "read_direct", recording_read_direct)
    return sizes


def plan_read(dataset: h5py.Dataset, index) -> np.ndarray:
    return read_with_plan(dataset, build_plans({"column": dataset}, index)["column"])


def row_bytes(dataset: h5py.Dataset) -> int:
    return int(np.prod(dataset.shape[1:], dtype=np.int64)) * dataset.dtype.itemsize


def set_limit_rows(monkeypatch, dataset: h5py.Dataset, rows: int) -> None:
    monkeypatch.setattr(plan_module, "MAX_READ_BUFFER", rows * row_bytes(dataset))


SIMPLE_INDICES = {
    "sorted-sparse": np.arange(5, N_ROWS, 37, dtype=np.int64),
    "unordered-duplicates": np.random.default_rng(0).integers(
        0, N_ROWS, 2_000, dtype=np.int64
    ),
}

CHUNKED_INDICES = {
    # Permuted chunks, as produced by sorting a structure collection
    "permuted": (
        np.array([9_000, 10, 4_000, 2_500, 7_700], dtype=np.int64),
        np.array([500, 40, 300, 1_000, 25], dtype=np.int64),
    ),
    "overlapping-nested-duplicate": (
        np.array([100, 120, 100, 5_000, 5_010, 9_990], dtype=np.int64),
        np.array([50, 10, 50, 400, 30, 10], dtype=np.int64),
    ),
}


@pytest.mark.parametrize("name", ["flat", "wide"])
@pytest.mark.parametrize("index", SIMPLE_INDICES.values(), ids=SIMPLE_INDICES)
def test_simple_read_respects_limit(columns, name, index, monkeypatch, read_sizes):
    dataset = columns[name]
    expected = dataset[:][index]
    set_limit_rows(monkeypatch, dataset, 100)

    result = plan_read(dataset, index)

    np.testing.assert_array_equal(result, expected)
    assert len(read_sizes) > 1
    assert max(read_sizes) <= 100


@pytest.mark.parametrize("name", ["flat", "wide"])
@pytest.mark.parametrize("index", CHUNKED_INDICES.values(), ids=CHUNKED_INDICES)
def test_chunked_read_respects_limit(columns, name, index, monkeypatch, read_sizes):
    dataset = columns[name]
    expected = dataset[:][into_array(index)]
    set_limit_rows(monkeypatch, dataset, 300)

    result = plan_read(dataset, index)

    np.testing.assert_array_equal(result, expected)
    assert len(read_sizes) > 1
    # A requested chunk larger than the limit is read on its own
    assert max(read_sizes) <= max(300, int(index[1].max()))


@pytest.mark.parametrize("name", ["flat", "wide"])
def test_simple_read_below_limit_is_one_span(columns, name, monkeypatch, read_sizes):
    dataset = columns[name]
    index = SIMPLE_INDICES["unordered-duplicates"]
    set_limit_rows(monkeypatch, dataset, N_ROWS)

    result = plan_read(dataset, index)

    np.testing.assert_array_equal(result, dataset[:][index])
    assert read_sizes == [int(index.max() - index.min() + 1)]


@pytest.mark.parametrize("name", ["flat", "wide"])
def test_limit_smaller_than_one_row(columns, name, monkeypatch, read_sizes):
    dataset = columns[name]
    index = np.array([7_000, 3, 3, 512, 9_999], dtype=np.int64)
    monkeypatch.setattr(plan_module, "MAX_READ_BUFFER", row_bytes(dataset) - 1)

    result = plan_read(dataset, index)

    np.testing.assert_array_equal(result, dataset[:][index])
    assert set(read_sizes) == {1}


def test_large_contiguous_chunk_reads_in_place(columns, monkeypatch, read_sizes):
    dataset = columns["wide"]
    index = (
        np.array([200, 900], dtype=np.int64),
        np.array([700, 5_000], dtype=np.int64),
    )
    set_limit_rows(monkeypatch, dataset, 300)
    allocations: list[tuple[int, ...]] = []
    original_empty = np.empty

    def recording_empty(shape, *args, **kwargs):
        allocations.append(tuple(np.atleast_1d(shape)))
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(plan_module.np, "empty", recording_empty)
    result = plan_read(dataset, index)

    np.testing.assert_array_equal(result, dataset[:][into_array(index)])
    # Oversized ranges are never split, and only the output is allocated
    assert read_sizes == [700, 5_000]
    assert allocations == [(5_700, 3)]


def test_oversized_range_does_not_size_buffer(columns, monkeypatch):
    dataset = columns["wide"]
    index = (
        np.array([20, 5_000, 0, 8_000], dtype=np.int64),
        np.array([5, 3_000, 10, 10], dtype=np.int64),
    )
    set_limit_rows(monkeypatch, dataset, 100)
    allocations: list[tuple[int, ...]] = []
    original_empty = np.empty

    def recording_empty(shape, *args, **kwargs):
        allocations.append(tuple(np.atleast_1d(shape)))
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(plan_module.np, "empty", recording_empty)
    result = plan_read(dataset, index)

    np.testing.assert_array_equal(result, dataset[:][into_array(index)])
    # Output, then one buffer sized by the small runs alone
    assert allocations[0] == (3_025, 3)
    assert all(shape[0] <= 100 for shape in allocations[1:])


def execute_plan(source: np.ndarray, plan, length: int) -> tuple[np.ndarray, list]:
    run_starts, run_sizes, run_segments, seg_out, seg_local, seg_sizes = plan
    output = np.full(length, -1, dtype=source.dtype)
    served = []
    first = 0
    for start, size, last in zip(
        run_starts, run_sizes, np.cumsum(run_segments), strict=True
    ):
        buffer = source[start : start + size]
        sizes = seg_sizes[first:last]
        served.append((int(size), int(sizes.sum()), int(last - first)))
        assert np.all(seg_local[first:last] + sizes <= size)
        output[into_array((seg_out[first:last], sizes))] = buffer[
            into_array((seg_local[first:last], sizes))
        ]
        first = last
    return output, served


PLAN_INDICES = {
    **SIMPLE_INDICES,
    **CHUNKED_INDICES,
    "repeated-row": np.full(1_000, 4_321, dtype=np.int64),
    "nested-in-oversized": (
        np.array([0, 50, 120], dtype=np.int64),
        np.array([1_000, 10, 900], dtype=np.int64),
    ),
}


@pytest.mark.parametrize("hdf5_chunk", [None, HDF5_CHUNK])
@pytest.mark.parametrize("max_rows", [1, 7, 100, N_ROWS])
@pytest.mark.parametrize("index", PLAN_INDICES.values(), ids=PLAN_INDICES)
def test_read_plan_is_bounded(index, max_rows, hdf5_chunk):
    source = np.arange(N_ROWS, dtype=np.int64)
    if isinstance(index, np.ndarray):
        plan = planlib.get_simple_read_plan(index, hdf5_chunk, max_rows)
    else:
        plan = planlib.get_chunked_read_plan(*index, hdf5_chunk, max_rows)

    flat = into_array(index)
    output, served = execute_plan(source, plan, len(flat))

    np.testing.assert_array_equal(output, flat)
    for size, n_served, n_segments in served:
        # Runs over the limit hold exactly one whole range, read in place
        assert (size <= max_rows and n_served <= max_rows) or (
            n_segments == 1 and n_served == size
        )


def test_read_plan_rejects_negative_sizes():
    starts = np.array([0, 10], dtype=np.int64)
    sizes = np.array([5, -1], dtype=np.int64)
    with pytest.raises(ValueError, match="non-negative"):
        planlib.get_chunked_read_plan(starts, sizes)


@pytest.fixture
def mixed_columns(tmp_path: Path):
    """Columns that pairwise agree or disagree on HDF5 chunk rows and row width."""
    path = tmp_path / "mixed.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.arange(N_ROWS, dtype=np.float64), chunks=(64,))
        f.create_dataset("b", data=np.arange(N_ROWS, dtype=np.int64), chunks=(64,))
        f.create_dataset("c", data=np.arange(N_ROWS, dtype=np.float64), chunks=(128,))
        f.create_dataset("d", data=np.arange(N_ROWS, dtype=np.float32), chunks=None)
        f.create_dataset(
            "wide",
            data=np.arange(3 * N_ROWS, dtype=np.float64).reshape(-1, 3),
            chunks=(64, 3),
        )
    with h5py.File(path, "r") as f:
        yield {name: f[name] for name in ["a", "b", "c", "d", "wide"]}


def test_build_plans_shares_matching_columns(mixed_columns, monkeypatch):
    monkeypatch.setattr(plan_module, "MAX_READ_BUFFER", 800)
    index = SIMPLE_INDICES["unordered-duplicates"]

    plans = build_plans(mixed_columns, index)

    assert list(plans) == list(mixed_columns)
    # Same chunk rows and row budget share one plan
    assert plans["a"] is plans["b"]
    # Different chunk rows, no chunking, or a different row budget do not
    others = [plans["a"], plans["c"], plans["d"], plans["wide"]]
    assert len({id(plan) for plan in others}) == len(others)


@pytest.mark.parametrize(
    "index",
    [*SIMPLE_INDICES.values(), *CHUNKED_INDICES.values()],
    ids=[*SIMPLE_INDICES, *CHUNKED_INDICES],
)
def test_read_with_plan_matches_get_data(mixed_columns, index, monkeypatch):
    monkeypatch.setattr(plan_module, "MAX_READ_BUFFER", 800)

    plans = build_plans(mixed_columns, index)

    for name, column in mixed_columns.items():
        np.testing.assert_array_equal(
            read_with_plan(column, plans[name]), get_data(column, index)
        )
        np.testing.assert_array_equal(
            read_with_plan(column, plans[name]), column[:][into_array(index)]
        )


@pytest.mark.parametrize(
    "index",
    [
        np.array([], dtype=np.int64),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
    ],
    ids=["simple", "chunked"],
)
def test_empty_read_keeps_row_shape(mixed_columns, index):
    plans = build_plans(mixed_columns, index)

    for name, column in mixed_columns.items():
        result = read_with_plan(column, plans[name])
        assert result.shape == (0,) + column.shape[1:]
        assert result.dtype == column.dtype


def test_hdf5_handler_plans_once_per_read(tmp_path: Path, monkeypatch):
    path = tmp_path / "handler.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for name in ["x", "y", "z"]:
            data.create_dataset(name, data=np.arange(N_ROWS, dtype=np.float64))
    calls: list[list[str]] = []
    original = handler_module.build_plans

    def recording_build_plans(columns, index):
        calls.append(list(columns))
        return original(columns, index)

    monkeypatch.setattr(handler_module, "build_plans", recording_build_plans)
    index = SIMPLE_INDICES["unordered-duplicates"]
    with h5py.File(path, "r") as f:
        handler = Hdf5Handler(f["data"], {"x": None, "y": None, "z": None}, index)
        result = handler.get_data(["z", "x", "y"])

        assert calls == [["z", "x", "y"]]
        assert list(result) == ["z", "x", "y"]
        for name, values in result.items():
            np.testing.assert_array_equal(values, f["data"][name][:][index])


@pytest.fixture
def handler_file(tmp_path: Path):
    path = tmp_path / "writer.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.create_dataset("x", data=np.arange(N_ROWS, dtype=np.float64))
        data.create_dataset(
            "wide", data=np.arange(3 * N_ROWS, dtype=np.int32).reshape(-1, 3)
        )
    with h5py.File(path, "r") as f:
        yield f


@pytest.fixture
def planned_reads(monkeypatch) -> list[int]:
    """Record the output length of every planned read issued by a writer source."""
    lengths: list[int] = []
    original = writer_module.read_with_plan

    def recording_read_with_plan(data, plan):
        lengths.append(plan.length)
        return original(data, plan)

    monkeypatch.setattr(writer_module, "read_with_plan", recording_read_with_plan)
    return lengths


def test_make_schema_sources_read_with_plans(handler_file, planned_reads):
    index = SIMPLE_INDICES["unordered-duplicates"]
    handler = Hdf5Handler(handler_file["data"], {"x": None, "wide": None}, index)

    schema = handler.make_schema(["x", "wide"])

    for name, writer in schema.columns.items():
        (source,) = writer.sources
        np.testing.assert_array_equal(source.data, handler_file["data"][name][:][index])
    assert planned_reads == [len(index), len(index)]


def test_reindexed_sources_drop_plan(handler_file, planned_reads):
    index = SIMPLE_INDICES["sorted-sparse"]
    handler = Hdf5Handler(handler_file["data"], {"wide": None}, index)
    (source,) = handler.make_schema(["wide"]).columns["wide"].sources
    expected = handler_file["data"]["wide"][:]

    reorder = np.arange(len(index))[::-1].copy()
    reordered = source.reorder(reorder)
    reindexed = source.with_index(SIMPLE_INDICES["unordered-duplicates"])

    np.testing.assert_array_equal(reordered.data, expected[index[reorder]])
    np.testing.assert_array_equal(
        reindexed.data, expected[SIMPLE_INDICES["unordered-duplicates"]]
    )
    assert planned_reads == []


def test_source_rejects_mismatched_plan(handler_file):
    dataset = handler_file["data"]["x"]
    plan = build_plans({"x": dataset}, SIMPLE_INDICES["sorted-sparse"])["x"]

    with pytest.raises(ValueError, match="does not match"):
        Hdf5Source(dataset, SIMPLE_INDICES["unordered-duplicates"], plan)
