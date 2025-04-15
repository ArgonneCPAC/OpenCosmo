import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def input_path(data_path):
    return data_path / "haloproperties.hdf5"


@pytest.mark.skip("Trees are not fully implemented yet")
def test_filter_write(input_path, tmp_path):
    tmp_file = tmp_path / "filtered_data.hdf5"
    with oc.open(input_path) as f:
        ds = f.filter(oc.col("fof_halo_mass") > 1e12)
        oc.write(tmp_file, ds)
        size_unfiltered = len(f.data)

    ds = oc.read(tmp_file)
    starts = ds._Dataset__handler._InMemoryHandler__tree._Tree__starts
    sizes = ds._Dataset__handler._InMemoryHandler__tree._Tree__sizes
    size = len(ds.data)
    assert size < size_unfiltered

    def is_valid(sl, size):
        return sl.stop > sl.start and sl.start >= 0 and sl.stop <= size

    for level in range(len(starts)):
        slice_total = np.sum(sizes[level])

        assert slice_total == size
        assert np.all(np.cumsum(np.insert(sizes[level], 0, 0))[:-1] == starts[level])
