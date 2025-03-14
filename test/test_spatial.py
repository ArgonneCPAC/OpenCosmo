import pytest

import opencosmo as oc


@pytest.fixture
def input_path(data_path):
    return data_path / "haloproperties.hdf5"


def test_filter_write(input_path, tmp_path):
    tmp_file = tmp_path / "filtered_data.hdf5"
    with oc.open(input_path) as f:
        ds = f.filter(oc.col("fof_halo_mass") > 1e12)
        oc.write(tmp_file, ds)
        size_unfiltered = len(f.data)

    ds = oc.read(tmp_file)
    slices = ds._Dataset__handler._InMemoryHandler__tree._Tree__slices
    size = len(ds.data)
    assert size < size_unfiltered

    def is_valid(sl, size):
        return sl.stop > sl.start and sl.start >= 0 and sl.stop <= size

    for level in slices:
        slice_total = sum((s.stop - s.start) for s in slices[level].values())
        assert slice_total == size
        assert all(is_valid(s, size) for s in slices[level].values())
