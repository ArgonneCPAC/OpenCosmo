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
    ds = ds.insert(fof_halo_px=derived)
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
    ds = ds.insert(fof_halo_px=derived)
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
    ds = ds.insert(fof_halo_p_sqr=derived)
    data = ds.data
    assert "fof_halo_p_sqr" in data.columns
    assert (
        data["fof_halo_p_sqr"].unit
        == data["fof_halo_mass"].unit
        * data["fof_halo_com_vx"].unit
        * data["fof_halo_com_vy"].unit
    )
    assert np.all(
        np.isclose(
            data["fof_halo_p_sqr"].value,
            (
                data["fof_halo_mass"].value
                * data["fof_halo_com_vx"].value
                * data["fof_halo_com_vy"].value
            ),
        )
    )


def test_scalars(input_path):
    ds = oc.open(input_path)
    derived1 = oc.col("fof_halo_mass") * 5
    derived2 = 3.0 * oc.col("fof_halo_mass")
    derived3 = 1 / oc.col("fof_halo_mass")
    derived4 = oc.col("fof_halo_mass") / 2.0
    ds = ds.insert(
        derived1=derived1, derived2=derived2, derived3=derived3, derived4=derived4
    )

    data = ds.data
    assert np.all(data["derived1"] == data["fof_halo_mass"] * 5)
    assert np.all(data["derived2"] == data["fof_halo_mass"] * 3)
    assert np.all(data["derived3"] == 1 / data["fof_halo_mass"])
    assert np.all(data["derived4"] == data["fof_halo_mass"] / 2)


def test_power(input_path):
    ds = oc.open(input_path)
    total_speed = (
        oc.col("fof_halo_com_vx") ** 2
        + oc.col("fof_halo_com_vy") ** 2
        + oc.col("fof_halo_com_vz") ** 2
    ) ** 0.5
    ke = 0.5 * oc.col("fof_halo_mass") * total_speed**2
    ds = ds.insert(ke=ke)
    data = ds.data
    assert (
        data["ke"].unit
        == data["fof_halo_mass"].unit * data["fof_halo_com_vx"].unit ** 2
    )
    assert all(
        np.isclose(
            data["ke"],
            0.5
            * data["fof_halo_mass"]
            * (
                data["fof_halo_com_vx"] ** 2
                + data["fof_halo_com_vy"] ** 2
                + data["fof_halo_com_vz"] ** 2
            ),
        )
    )


def test_derive_unit_change(input_path):
    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.insert(fof_halo_px=derived)
    comoving_data = ds.data
    comoving_unit = comoving_data["fof_halo_px"].unit
    ds = ds.with_units("scalefree")
    scalefree_data = ds.data
    scalefree_unit = scalefree_data["fof_halo_px"].unit
    assert comoving_unit != scalefree_unit
    assert comoving_unit == (
        comoving_data["fof_halo_mass"].unit * comoving_data["fof_halo_com_vx"].unit
    )
    assert scalefree_unit == (
        scalefree_data["fof_halo_mass"].unit * scalefree_data["fof_halo_com_vx"].unit
    )
    assert not np.any(
        comoving_data["fof_halo_px"].value == scalefree_data["fof_halo_px"].value
    )


def test_derive_mask(input_path):
    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.insert(fof_halo_px=derived)
    comoving_data = ds.data["fof_halo_px"]
    val = 0.5 * (comoving_data.max() - comoving_data.min()) + comoving_data.min()
    ds_masked = ds.filter(oc.col("fof_halo_px") > val)
    assert all(ds_masked.data["fof_halo_px"].value > val)


def test_derive_children(input_path):
    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.insert(fof_halo_px=derived)
    derived2 = 0.5 * oc.col("fof_halo_px") * oc.col("fof_halo_com_vx")
    ds = ds.insert(derived2=derived2)
    data = ds.data
    assert np.all(
        np.isclose(
            data["derived2"], 0.5 * data["fof_halo_mass"] * data["fof_halo_com_vx"] ** 2
        )
    )


def test_derive_children_select(input_path):
    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.insert(fof_halo_px=derived)
    derived2 = 0.5 * oc.col("fof_halo_px") * oc.col("fof_halo_com_vx")
    ds = ds.insert(derived2=derived2)
    derived_data = ds.data["derived2"]

    to_select = ["fof_halo_com_vy", "derived2"]
    ds = ds.select(to_select)
    data = ds.data
    assert set(data.columns) == set(to_select)
    assert np.all(derived_data == data["derived2"])
