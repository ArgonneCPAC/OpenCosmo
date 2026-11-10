import shutil

import astropy.units as u
import h5py
import numpy as np
import opencosmo as oc
import pytest
from astropy.cosmology import units as cu

HYDRO_COMPONENTS = {"total", "cdm", "bar", "gas", "star", "sfgas", "wind", "agn"}


@pytest.fixture
def hydro(test_data):
    return oc.open(test_data.power_spectrum.hydro)


@pytest.fixture
def hydro_copy(test_data, tmp_path):
    path = tmp_path / "hydro.hdf5"
    shutil.copy(test_data.power_spectrum.hydro, path)
    return path


def test_open(hydro):
    assert isinstance(hydro, oc.PowerSpectrum)
    assert set(hydro.components) == HYDRO_COMPONENTS
    n_steps = len(hydro.steps)
    n_k = len(hydro.n_modes)
    assert np.all(np.diff(hydro.steps) > 0)
    assert hydro.redshifts.shape == (n_steps,)
    assert hydro.k.shape == (n_k,)
    assert np.all(np.diff(hydro.k) > 0)
    for component in hydro.components:
        assert hydro.get_data(component).shape == (n_steps, n_k)


def test_open_gravity_only(test_data):
    ps = oc.open(test_data.power_spectrum.gravity_only)
    assert ps.components == ("total",)
    assert ps.get_data().shape == (len(ps.steps), len(ps.k))


def test_header(hydro):
    assert hydro.header.file.data_type == "power_spectrum"
    assert hydro.header.file.step is None
    assert hydro.power_spectrum["grid"] == hydro.simulation["pm_grid"]
    assert hydro.power_spectrum["binning"] == "linear"
    assert hydro.cosmology.h > 0


def test_comoving_units(hydro):
    assert hydro.k.unit == u.Mpc**-1
    assert hydro.get_data("cdm").unit == u.Mpc**3


def test_scalefree_units(hydro):
    scalefree = hydro.with_units("scalefree")
    h = hydro.cosmology.h
    assert scalefree.k.unit == cu.littleh / u.Mpc
    assert scalefree.get_data().unit == (u.Mpc / cu.littleh) ** 3
    assert np.allclose(scalefree.k.value * h, hydro.k.value)
    assert np.allclose(scalefree.get_data().value / h**3, hydro.get_data().value)


def test_physical_units(hydro):
    physical = hydro.with_units("physical")
    a = 1 / (1 + hydro.redshifts)
    k = physical.k
    assert k.shape == (len(hydro.steps), len(hydro.k))
    assert np.allclose(k.value, hydro.k.value / a[:, np.newaxis])
    assert np.allclose(
        physical.get_data().value, hydro.get_data().value * a[:, np.newaxis] ** 3
    )
    step = hydro.steps[0]
    assert np.allclose(
        physical.get_data(step=step).value,
        hydro.get_data(step=step).value * a[0] ** 3,
    )
    # The header has no single redshift, so it stays comoving
    assert physical.simulation["box_size"].unit == u.Mpc


def test_unitless(hydro):
    unitless = hydro.with_units("unitless")
    assert not isinstance(unitless.k, u.Quantity)
    assert not isinstance(unitless.get_data(), u.Quantity)


def test_with_units_returns_new_object(hydro):
    scalefree = hydro.with_units("scalefree")
    assert scalefree is not hydro
    assert hydro.k.unit == u.Mpc**-1


def test_get_step(hydro):
    full = hydro.get_data("cdm")
    for i, step in enumerate(hydro.steps):
        assert np.array_equal(hydro.get_data("cdm", step=step), full[i])


def test_get_redshift(hydro):
    i = len(hydro.steps) // 2
    redshift = hydro.redshifts[i] + 1e-3
    assert np.array_equal(hydro.get_data(redshift=redshift), hydro.get_data()[i])


def test_unknown_component(hydro):
    with pytest.raises(ValueError, match="no component 'dm'"):
        hydro.get_data("dm")


def test_unknown_step(hydro):
    with pytest.raises(ValueError, match="Step 1 is not in this power spectrum"):
        hydro.get_data(step=1)


def test_step_and_redshift(hydro):
    with pytest.raises(ValueError, match="either a step or a redshift"):
        hydro.get_data(step=hydro.steps[0], redshift=0)


def test_missing_spectrum_is_nan(hydro_copy):
    with h5py.File(hydro_copy, "r+") as f:
        f["power_spectrum/pk_agn"][0] = np.nan
    ps = oc.open(hydro_copy)
    assert np.all(np.isnan(ps.get_data("agn", step=ps.steps[0])))
    assert not np.any(np.isnan(ps.get_data("agn")[1:]))


def test_missing_axis(hydro_copy):
    with h5py.File(hydro_copy, "r+") as f:
        del f["power_spectrum/n_modes"]
    with pytest.raises(ValueError, match="missing the datasets n_modes"):
        oc.open(hydro_copy)


def test_wrong_shape(hydro_copy):
    with h5py.File(hydro_copy, "r+") as f:
        pk = f["power_spectrum/pk_cdm"][:]
        del f["power_spectrum/pk_cdm"]
        f["power_spectrum/pk_cdm"] = pk[:, :-1]
    with pytest.raises(ValueError, match="pk_cdm must have the shape"):
        oc.open(hydro_copy)


def test_open_with_other_simulation(test_data):
    with pytest.raises(TypeError, match="Simulation collection only accepts"):
        oc.open(test_data.power_spectrum.hydro, test_data.power_spectrum.gravity_only)
