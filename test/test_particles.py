import astropy.units as u
import h5py
import pytest
from astropy.cosmology import units as cu

import opencosmo as oc


@pytest.fixture
def input_path(data_path):
    return data_path / "haloparticles.hdf5"


def test_read_particles(input_path):
    dc = oc.read(input_path)
    assert isinstance(dc, oc.DataCollection)


def test_read_one_particle(input_path):
    ds = oc.read(input_path, "star_particles")
    assert isinstance(ds, oc.Dataset)


def test_read_some_particles(input_path):
    keys = ["star_particles", "dm_particles"]
    ds = oc.read(input_path, keys)
    assert isinstance(ds, oc.DataCollection)
    assert len(ds) == 2
    assert "star_particles" in ds
    assert "dm_particles" in ds


def test_read_all_particles(input_path):
    with h5py.File(input_path, "r") as f:
        keys = list(f.keys())

    ds = oc.read(input_path)
    assert isinstance(ds, oc.DataCollection)
    assert len(ds) == len(keys) - 1


def test_read_invalid_dataset(input_path):
    with pytest.raises(ValueError):
        oc.read(input_path, "invalid_particles")


def test_read_invalid_datasets(input_path):
    with pytest.raises(ValueError):
        oc.read(input_path, ["star_particles", "invalid_particles"])


def test_age_h0_units(input_path, tmp_path):
    dc = oc.read(input_path, "agn_particles")
    assert dc.data["age"].unit == u.yr
    dc = dc.with_units("scalefree")
    assert dc.data["age"].unit == u.yr / cu.littleh


def test_write_particles(input_path, tmp_path):
    dc = oc.read(input_path)
    new_path = tmp_path / "haloparticles.hdf5"
    oc.write(new_path, dc)
    new_dc = oc.read(new_path)
    for key in dc.keys():
        assert key in new_dc
        assert all(dc[key].data == new_dc[key].data)
