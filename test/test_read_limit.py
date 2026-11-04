from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import opencosmo.index.get as get_module
import pytest
from opencosmo.index import get_data, into_array

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


def row_bytes(dataset: h5py.Dataset) -> int:
    return int(np.prod(dataset.shape[1:], dtype=np.int64)) * dataset.dtype.itemsize


def set_limit_rows(monkeypatch, dataset: h5py.Dataset, rows: int) -> None:
    monkeypatch.setattr(get_module, "MAX_SINGLE_READ", rows * row_bytes(dataset))


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

    result = get_data(dataset, index)

    np.testing.assert_array_equal(result, expected)
    assert len(read_sizes) > 1
    assert max(read_sizes) <= 100


@pytest.mark.parametrize("name", ["flat", "wide"])
@pytest.mark.parametrize("index", CHUNKED_INDICES.values(), ids=CHUNKED_INDICES)
def test_chunked_read_respects_limit(columns, name, index, monkeypatch, read_sizes):
    dataset = columns[name]
    expected = dataset[:][into_array(index)]
    set_limit_rows(monkeypatch, dataset, 300)

    result = get_data(dataset, index)

    np.testing.assert_array_equal(result, expected)
    assert len(read_sizes) > 1
    # A single requested chunk larger than the limit is read on its own
    assert max(read_sizes) <= max(300, int(index[1].max()))


@pytest.mark.parametrize("name", ["flat", "wide"])
def test_simple_read_below_limit_is_one_span(columns, name, monkeypatch, read_sizes):
    dataset = columns[name]
    index = SIMPLE_INDICES["unordered-duplicates"]
    set_limit_rows(monkeypatch, dataset, N_ROWS)

    result = get_data(dataset, index)

    np.testing.assert_array_equal(result, dataset[:][index])
    assert read_sizes == [int(index.max() - index.min() + 1)]


@pytest.mark.parametrize("name", ["flat", "wide"])
def test_limit_smaller_than_one_row(columns, name, monkeypatch, read_sizes):
    dataset = columns[name]
    index = np.array([7_000, 3, 3, 512, 9_999], dtype=np.int64)
    monkeypatch.setattr(get_module, "MAX_SINGLE_READ", row_bytes(dataset) - 1)

    result = get_data(dataset, index)

    np.testing.assert_array_equal(result, dataset[:][index])
    assert set(read_sizes) == {1}
