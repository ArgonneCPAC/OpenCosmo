import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def particle_path(snapshot_path):
    return snapshot_path / "haloparticles.hdf5"


def test_derive_multiply(input_path):
    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.update(fof_halo_px=derived)
    data = ds.data
    assert "fof_halo_px" in data.columns
    assert (
        data["fof_halo_px"].unit
        == data["fof_halo_mass"].unit * data["fof_halo_com_vx"].unit
    )
    assert np.all(
        data["fof_halo_px"].value
        == data["fof_halo_mass"].value * data["fof_halo_com_vx"].value
    )


def test_derive_divide(input_path):
    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") / oc.col("fof_halo_com_vx")
    ds = ds.update(fof_halo_px=derived)
    data = ds.data
    assert "fof_halo_px" in data.columns
    assert (
        data["fof_halo_px"].unit
        == data["fof_halo_mass"].unit / data["fof_halo_com_vx"].unit
    )
    assert np.all(
        data["fof_halo_px"].value
        == data["fof_halo_mass"].value / data["fof_halo_com_vx"].value
    )


def test_derive_chain(input_path):
    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") * (
        oc.col("fof_halo_com_vx") * oc.col("fof_halo_com_vy")
    )
    ds = ds.update(fof_halo_px_sqr=derived)
    data = ds.data
    print(data)

    assert False
