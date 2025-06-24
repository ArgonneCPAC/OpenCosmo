from pathlib import Path
from shutil import copyfile

import astropy.cosmology.units as cu
import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.table import Column

import opencosmo as oc


def add_column(tmp_path: Path, original_file: Path, column: Column):
    new_path = tmp_path / original_file.name
    copyfile(original_file, new_path)
    with h5py.File(new_path, "r+") as file:
        data = file["data"]
        data.create_dataset(column.name, data=column.data)
        if column.unit is not None:
            data[column.name].attrs["unit"] = str(column.unit)
    return new_path


@pytest.fixture
def haloproperties_step_path(snapshot_path):
    return snapshot_path / "haloproperties_step310.hdf5"


@pytest.fixture
def haloproperties_lc_path(lightcone_path):
    return lightcone_path / "step_600" / "haloproperties.hdf5"


@pytest.fixture
def galaxy_input_path(snapshot_path):
    return snapshot_path / "galaxyproperties.hdf5"


@pytest.fixture
def halo_input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def sod_particle_input_path(snapshot_path):
    return snapshot_path / "sodbighaloparticles.hdf5"


@pytest.fixture
def sod_properties_input_path(snapshot_path):
    return snapshot_path / "sodproperties.hdf5"


@pytest.fixture
def input_path(request):
    return request.getfixturevalue(request.param)


pytestmark = pytest.mark.parametrize(
    "input_path",
    [
        "galaxy_input_path",
        "halo_input_path",
        "sod_properties_input_path",
    ],
    indirect=True,
)


def test_physcal_units(haloproperties_step_path, input_path):
    # need a redshift != 0 test set
    ds = oc.open(haloproperties_step_path)

    ds_physical = ds.with_units("physical")

    data_physical = ds_physical.data
    data = ds.data
    cols = data.columns
    z = ds.redshift

    position_cols = filter(lambda col: col.split("_")[-1] in ["x", "y", "z"], cols)
    position_cols = filter(lambda col: "angmom" not in col, position_cols)
    assert z != 0
    for col in position_cols:
        assert data_physical[col].unit == u.Mpc
        assert data[col].unit == u.Mpc
        assert np.all(
            data_physical[col].value == data[col].value * ds.cosmology.scale_factor(z)
        )


def test_logarithmic_units(input_path):
    dataset = oc.open(input_path)
    data = dataset.data
    cols = data.columns
    for col in cols:
        if "log" in col.lower():
            assert isinstance(data[col].unit, u.DexUnit)


def test_parse_velocities(input_path):
    dataset = oc.open(input_path)
    data = dataset.data
    cols = data.columns
    velocity_cols = list(
        filter(lambda col: col.upper()[-2:] in ["VX", "VY", "VZ"], cols)
    )
    for col in velocity_cols:
        assert data[col].unit == u.km / u.s


def test_comoving_vs_scalefree(input_path):
    comoving = oc.open(input_path).with_units("comoving")
    scalefree = oc.open(input_path).with_units("scalefree")
    cols = comoving.data.columns
    position_cols = filter(lambda col: col.split("_")[-1] in ["x", "y", "z"], cols)
    position_cols = filter(lambda col: "angmom" not in col, position_cols)
    h = comoving.cosmology.h
    for col in position_cols:
        assert scalefree.data[col].unit == u.Mpc / cu.littleh
        assert comoving.data[col].unit == u.Mpc
        assert np.all(
            np.isclose(comoving.data[col].value, scalefree.data[col].value / h)
        )


def test_angular_momentum(input_path):
    dataset = oc.open(input_path)
    dataset = dataset.with_units("scalefree")
    data = dataset.data
    cols = data.columns
    angmom_cols = filter(lambda col: "angmom" in col, cols)
    angmom_unit = (u.Msun / cu.littleh) * (u.km / u.s) * (u.Mpc / cu.littleh)
    for col in angmom_cols:
        assert data[col].unit == angmom_unit


def test_parse_positions(input_path):
    dataset = oc.open(input_path).with_units("scalefree")
    data = dataset.data
    cols = data.columns
    position_cols = filter(lambda col: col.split("_")[-1] in ["x", "y", "z"], cols)
    position_cols = filter(lambda col: "angmom" not in col, position_cols)
    for col in position_cols:
        assert data[col].unit == u.Mpc / cu.littleh


def test_parse_mass(input_path):
    dataset = oc.open(input_path).with_units("scalefree")
    data = dataset.data
    cols = data.columns
    mass_cols = filter(
        lambda col: any(part == "mass" or part[0] == "M" for part in col.split("_")),
        cols,
    )
    mass_cols = filter(lambda col: "_c_" not in col, mass_cols)
    for col in mass_cols:
        assert data[col].unit == u.Msun / cu.littleh


def test_data_update_doesnt_propogate(input_path):
    dataset = oc.open(input_path)
    data = dataset.data
    new_column = np.random.rand(len(data))
    data["new_column"] = new_column
    assert "new_column" in data.columns
    assert "new_column" not in dataset.data.columns


def test_unitless_convention(input_path):
    dataset = oc.open(input_path).with_units("unitless")
    data = dataset.data
    cols = data.columns
    assert all(data[col].unit is None for col in cols)


def test_unit_conversion(input_path):
    dataset = oc.open(input_path).with_units("unitless")
    data = dataset.data
    cols = data.columns

    unitful_dataset = dataset.with_units("comoving")
    position_cols = filter(lambda col: col.split("_")[-1] in ["x", "y", "z"], cols)
    position_cols = filter(lambda col: "angmom" not in col, position_cols)

    unitless_data = dataset.data
    unitful_data = unitful_dataset.data
    for col in position_cols:
        assert unitful_data[col].unit == u.Mpc
    for col in cols:
        assert unitless_data[col].unit is None

    converted_unitless = unitful_dataset.with_units("unitless")
    converted_unitless_data = converted_unitless.data
    for col in cols:
        assert converted_unitless_data[col].unit is None


def test_invalid_unit_convention(input_path):
    with pytest.raises(ValueError):
        oc.open(input_path).with_units("invalid_unit")
