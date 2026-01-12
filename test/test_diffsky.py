import astropy.units as u
import numpy as np
import pytest

import opencosmo as oc
from opencosmo.column import add_mag_cols


@pytest.fixture
def core_path_487(diffsky_path):
    return diffsky_path / "lj_487.hdf5"


@pytest.fixture
def core_path_475(diffsky_path):
    return diffsky_path / "lj_475.hdf5"


@pytest.fixture
def invalid_data_path(diffsky_path):
    return diffsky_path / "random_data.hdf5"


def test_comoving_to_physical(core_path_487):
    cores = oc.open(core_path_487, synth_cores=True).select(["redshift_true", "x"])
    data_physical = cores.with_units("physical").select(["redshift_true", "x"]).data
    data_comoving = cores.select(["redshift_true", "x"]).data
    a = 1 / (data_physical["redshift_true"] + 1)
    assert np.all(np.isclose(data_physical["x"], data_comoving["x"] * a))


def test_comoving_to_scalefree(core_path_487):
    with pytest.raises(ValueError):
        _ = oc.open(core_path_487, synth_cores=True).with_units("scalefree")


def test_comoving_to_unitless(core_path_487):
    ds = oc.open(core_path_487, synth_cores=True)
    data = ds.data
    data_unitless = ds.with_units("unitless").data
    for col in data.columns:
        assert np.all(data[col].value == data_unitless[col].value)


def test_filter_take(core_path_475, core_path_487):
    ds = oc.open(core_path_475, core_path_487)
    ds = ds.filter(oc.col("logsm_obs") < 13, oc.col("logsm_obs") > 10)
    ds = ds.select("logsm_obs")
    assert ds.data.max().value < 13 and ds.data.min().value > 10
    ds = ds.take(10)
    assert ds.data.max().value < 13 and ds.data.min().value > 10


def test_open_multiple_write(core_path_487, core_path_475, tmp_path):
    ds = oc.open(core_path_487, core_path_475)
    original_length = len(ds)
    original_redshift_range = ds.z_range
    output = tmp_path / "synth_gals.hdf5"
    oc.write(output, ds)
    ds = oc.open(output)
    assert len(ds) == original_length
    assert ds.z_range == original_redshift_range


def test_cone_search(core_path_475, core_path_487):
    center = (40, 67)
    radius = 2
    ds = oc.open(core_path_487, core_path_475, synth_cores=True)
    region = oc.make_cone(center, radius)
    ds = ds.bound(region)


def test_repr(core_path_475, core_path_487):
    ds = oc.open(core_path_487, core_path_475)
    ds = ds.with_redshift_range(0, 0.1)
    ds = ds.select(["ra", "dec"])
    assert str(ds)


def test_open_write_with_synthetics(core_path_475, core_path_487, tmp_path):
    ds = oc.open(core_path_487, core_path_475, synth_cores=True)
    print(len(ds))
    ds = ds.filter(oc.col("lsst_g") < 20).take(10_000)
    print(len(ds))
    oc.write(tmp_path / "test.hdf5", ds)
    ds = oc.open(tmp_path / "test.hdf5")

    assert False


def test_add_logarithmic_units(core_path_487):
    ds = oc.open(core_path_487, synth_cores=True)
    log_sfr = oc.col("logsm_obs") + oc.col("logssfr_obs")
    ds = ds.with_new_columns(log_sfr=log_sfr)
    data = ds.select(("logsm_obs", "logssfr_obs", "log_sfr")).get_data()

    sfr = data["logssfr_obs"] + data["logsm_obs"]
    assert np.all(sfr == data["log_sfr"])


def test_add_two_non_logarithmic_units(core_path_487):
    ds = oc.open(core_path_487, synth_cores=True)
    log_sfr = oc.col("dec") + oc.col("x_nfw")
    with pytest.raises(oc.units.UnitsError):
        ds = ds.with_new_columns(log_sfr=log_sfr)


def test_mag_units(core_path_475, core_path_487):
    ds = oc.open(core_path_487, core_path_475)
    mag_columns = filter(lambda c: "lsst_" in c, ds.columns)
    mag_data = ds.select(mag_columns).get_data()
    for col in mag_data.itercols():
        assert col.unit == u.ABmag


def test_add_mag_units(core_path_475, core_path_487):
    ds = oc.open(core_path_487, core_path_475)
    mag_columns = list(filter(lambda c: "lsst_" in c, ds.columns))
    total_mag = add_mag_cols(*mag_columns)
    ds = ds.with_new_columns(lsst_total=total_mag)

    mag_columns += ["lsst_total"]

    total_mag = np.zeros(len(ds), dtype=float)
    data = ds.select(mag_columns).get_data("numpy")
    for column in mag_columns:
        if column == "lsst_total":
            continue
        total_mag += 10 ** (-0.4 * data[column])

    total_mag = -2.5 * np.log10(total_mag)

    assert np.all(data["lsst_total"] == total_mag)


def test_add_mag_units_unitless(core_path_475, core_path_487):
    ds = oc.open(core_path_487, core_path_475).with_units("unitless")
    mag_columns = list(filter(lambda c: "lsst_" in c, ds.columns))
    total_mag = add_mag_cols(*mag_columns)
    ds = ds.with_new_columns(lsst_total=total_mag)
    mag_columns += ["lsst_total"]

    data = ds.select(mag_columns).get_data("numpy")
    total_mag = np.zeros(len(ds), dtype=float)
    for column in mag_columns:
        if column == "lsst_total":
            continue
        total_mag += 10 ** (-0.4 * data[column])

    total_mag = -2.5 * np.log10(total_mag)
    assert np.all(data["lsst_total"] == total_mag)


def test_open_bad_data(core_path_475, core_path_487, invalid_data_path):
    with pytest.raises(ValueError, match=str(invalid_data_path)):
        oc.open(core_path_475, core_path_487, invalid_data_path)
