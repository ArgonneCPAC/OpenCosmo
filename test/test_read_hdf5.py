import io

import h5py
import hdf5plugin
import numpy as np
import opencosmo.index.get as get_module
import pytest
from opencosmo.handler.hdf5 import Hdf5Handler, _open_column
from opencosmo.index import get_data, into_array, read_columns, single_chunk

N_ROWS = 10_000
CHUNK_ROWS = 256


class CountingFile(io.RawIOBase):
    """A read-only file that counts the reads HDF5 issues against it."""

    def __init__(self, path):
        self.file = open(path, "rb")  # noqa: SIM115 (closed by close())
        self.reads = 0

    def readinto(self, buffer):
        self.reads += 1
        return self.file.readinto(buffer)

    def seek(self, *args):
        return self.file.seek(*args)

    def tell(self):
        return self.file.tell()

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        self.file.close()
        super().close()


@pytest.fixture(scope="module")
def columns_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("read") / "columns.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "chunked",
            data=rng.random(N_ROWS).astype(np.float32),
            chunks=(CHUNK_ROWS,),
            compression=hdf5plugin.Blosc2(),
        )
        f.create_dataset(
            "wide",
            data=rng.integers(0, 1000, (N_ROWS, 7)).astype(np.int32),
            chunks=(CHUNK_ROWS, 7),
            compression=hdf5plugin.Blosc2(),
        )
        f.create_dataset(
            "same_layout",
            data=rng.integers(0, 1000, N_ROWS).astype(np.int32),
            chunks=(CHUNK_ROWS,),
            compression=hdf5plugin.Blosc2(),
        )
        f.create_dataset(
            "half_chunks",
            data=rng.integers(0, 1000, N_ROWS).astype(np.int64),
            chunks=(CHUNK_ROWS // 2,),
            compression=hdf5plugin.Blosc2(),
        )
        f.create_dataset("contiguous", data=np.arange(N_ROWS, dtype=np.int64))
    return path


COLUMNS = ["chunked", "wide", "same_layout", "half_chunks", "contiguous"]


def random_chunked(rng, n_segments, mean_size):
    starts = np.sort(rng.choice(N_ROWS, n_segments, replace=False))
    sizes = rng.integers(1, 2 * mean_size, n_segments)
    sizes = np.minimum(starts + sizes, np.append(starts[1:], N_ROWS)) - starts
    return (starts.astype(np.int64), sizes.astype(np.int64))


def indices(seed):
    rng = np.random.default_rng(seed)
    chunked = random_chunked(rng, 200, 20)
    return {
        "full": (np.array([0], dtype=np.int64), np.array([N_ROWS], dtype=np.int64)),
        "chunked": chunked,
        "simple": into_array(chunked),
        "sparse_simple": np.sort(rng.choice(N_ROWS, 50, replace=False)),
        "unsorted_simple": rng.choice(N_ROWS, 500),
        "permuted_chunked": (chunked[0][::-1].copy(), chunked[1][::-1].copy()),
        "single_row": np.array([N_ROWS - 1], dtype=np.int64),
    }


@pytest.mark.parametrize("column", ["chunked", "wide", "contiguous"])
@pytest.mark.parametrize("seed", range(3))
def test_read_matches_fancy_indexing(columns_path, column, seed):
    with h5py.File(columns_path) as f:
        ds = f[column]
        data = ds[:]
        for name, index in indices(seed).items():
            result = get_data(ds, index)
            np.testing.assert_array_equal(result, data[into_array(index)], name)
            assert result.dtype == ds.dtype


def test_read_wide_column_from_test_data(test_data):
    with h5py.File(test_data.snapshot.primary.halo_profiles) as f:
        ds = f["data/sod_halo_bin_gas_ne"]
        data = ds[:]
        index = random_chunked(np.random.default_rng(1), 300, 5)
        index = (index[0][index[0] < len(ds) - 20], index[1][index[0] < len(ds) - 20])
        np.testing.assert_array_equal(get_data(ds, index), data[into_array(index)])


def test_read_of_empty_index_is_empty(columns_path):
    with h5py.File(columns_path) as f:
        assert len(get_data(f["chunked"], np.array([], dtype=np.int64))) == 0


def test_read_rejects_out_of_range_index(columns_path):
    with (
        h5py.File(columns_path) as f,
        pytest.raises(ValueError, match="out of range"),
    ):
        get_data(f["chunked"], np.array([N_ROWS], dtype=np.int64))


def test_each_touched_chunk_is_read_once(columns_path):
    # With no chunk cache, the old one-read-per-segment loop fetched each chunk
    # once per segment touching it.
    index = random_chunked(np.random.default_rng(2), 2000, 3)
    touched = len(np.unique(into_array(index) // CHUNK_ROWS))
    raw = CountingFile(columns_path)
    with h5py.File(raw, "r", rdcc_nbytes=0) as f:
        ds = f["chunked"]
        ds[:1]  # load the chunk index metadata
        raw.reads = 0
        get_data(ds, index)
        assert raw.reads <= touched + 2
    assert len(index[0]) > 4 * touched


def test_opened_column_caches_a_whole_chunk(columns_path):
    with h5py.File(columns_path, rdcc_nbytes=1024) as f:
        ds = _open_column(f, "wide")
        assert ds.id.get_access_plist().get_chunk_cache()[1] >= CHUNK_ROWS * 7 * 4


def test_opened_column_keeps_a_larger_cache(columns_path):
    with h5py.File(columns_path, rdcc_nbytes=64 << 20) as f:
        ds = _open_column(f, "chunked")
        assert ds.id.get_access_plist().get_chunk_cache()[1] == 64 << 20


def test_successive_small_reads_reuse_the_cached_chunk(columns_path):
    # Iterating over structures reads a few rows at a time from the same chunk.
    raw = CountingFile(columns_path)
    with h5py.File(raw, "r", rdcc_nbytes=1024) as f:
        ds = _open_column(f, "wide")
        ds[:1]
        raw.reads = 0
        for start in range(0, CHUNK_ROWS, 8):
            get_data(ds, single_chunk(CHUNK_ROWS + start, 8))
        assert raw.reads <= 2


@pytest.mark.parametrize("seed", range(3))
def test_read_columns_matches_single_column_reads(columns_path, seed):
    with h5py.File(columns_path) as f:
        columns = {name: f[name] for name in reversed(COLUMNS)}
        for name, index in indices(seed).items():
            result = read_columns(columns, index)
            assert list(result) == list(columns), name
            for column, values in result.items():
                np.testing.assert_array_equal(
                    values, get_data(columns[column], index), f"{name}/{column}"
                )
                assert values.dtype == columns[column].dtype


def test_read_columns_plans_once_per_layout(columns_path, monkeypatch):
    original_plan = get_module.plan
    planned = []

    def counting_plan(index, *layout):
        planned.append(layout)
        return original_plan(index, *layout)

    monkeypatch.setattr(get_module, "plan", counting_plan)
    with h5py.File(columns_path) as f:
        read_columns(
            {name: f[name] for name in COLUMNS},
            random_chunked(np.random.default_rng(3), 50, 10),
        )
    # "chunked" and "same_layout" share rows, chunk length and row size.
    assert len(planned) == len(COLUMNS) - 1
    assert len(set(planned)) == len(planned)


def test_read_columns_of_empty_index(columns_path):
    with h5py.File(columns_path) as f:
        result = read_columns(
            {name: f[name] for name in COLUMNS}, np.array([], dtype=np.int64)
        )
        for name, values in result.items():
            assert len(values) == 0
            assert values.dtype == f[name].dtype


def test_handler_reads_requested_columns_in_order(columns_path):
    index = random_chunked(np.random.default_rng(4), 100, 10)
    with h5py.File(columns_path) as f:
        handler = Hdf5Handler(f, {name: None for name in COLUMNS}, index)
        requested = ["half_chunks", "chunked", "wide"]
        result = handler.get_data(requested)
        assert list(result) == requested
        rows = into_array(index)
        for name in requested:
            np.testing.assert_array_equal(result[name], f[name][:][rows])
