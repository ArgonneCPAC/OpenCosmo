import pytest
import opencosmo as oc

@pytest.fixture
def multi_path(data_path):
    return data_path / "haloproperties_multi.hdf5"


def test_multi_filter(multi_path):
    collection = oc.read(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)

    for ds in collection:
        assert all(ds.data["sod_halo_mass"] > 0)

def test_multi_filter_write(multi_path, tmp_path):
    collection = oc.read(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)
    for ds in collection:
        assert all(ds.data["sod_halo_mass"] > 0)
    oc.write(tmp_path / "filtered.hdf5", collection)

    collection = oc.read(tmp_path / "filtered.hdf5")
    for ds in collection:
        assert all(ds.data["sod_halo_mass"] > 0)
    
