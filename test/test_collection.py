import pytest
from pathlib import Path

import opencosmo as oc


@pytest.fixture
def multi_path(data_path):
    return data_path / "haloproperties_multi.hdf5"


@pytest.fixture
def all_path(data_path: Path):
    files = ["haloparticles.hdf5", "haloproperties.hdf5", "sodproperties.hdf5"]

    hdf_files = [data_path / file for file in files]
    return list(hdf_files)

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

def test_data_linking(all_path):
    collection = oc.collection.collection.LinkedCollection.from_files(*all_path)
    print(collection)
    assert False

