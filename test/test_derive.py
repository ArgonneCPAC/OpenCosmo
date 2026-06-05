import astropy.units as u
import numpy as np
import pytest

import opencosmo as oc
from opencosmo.column import norm_cols, offset_3d
from opencosmo.units import UnitsError


@pytest.fixture
def properties_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def particles_path(snapshot_path):
    return snapshot_path / "haloparticles.hdf5"


def test_derive_multiply(properties_path):
    ds = oc.open(properties_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    data = ds.get_data()
    assert "fof_halo_px" in data.columns
    assert (
        data["fof_halo_px"].unit
        == data["fof_halo_mass"].unit * data["fof_halo_com_vx"].unit
    )
    assert np.all(
        np.isclose(
            data["fof_halo_px"].value,
            data["fof_halo_mass"].value * data["fof_halo_com_vx"].value,
        )
    )


def test_derive_addition(properties_path, particles_path, tmp_path):
    ds = oc.open(properties_path, particles_path)
    dr = offset_3d("fof_halo_com", "sod_halo_com")
    xoff = dr / oc.col("sod_halo_radius")

    ds = ds.with_new_columns("halo_properties", xoff=xoff)
    xoff = ds["halo_properties"].select("xoff").get_data()

    assert xoff.unit == u.dimensionless_unscaled


def test_derive_write(properties_path, tmp_path):
    ds = oc.open(properties_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    data = ds.get_data()
    oc.write(tmp_path / "test.hdf5", ds)
    written_data = oc.open(tmp_path / "test.hdf5").get_data()
    assert np.all(
        np.isclose(data["fof_halo_px"].value, written_data["fof_halo_px"].value)
    )


def test_derive_divide(properties_path):
    ds = oc.open(properties_path)
    derived = oc.col("fof_halo_mass") / oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    data = ds.get_data()
    assert "fof_halo_px" in data.columns
    assert (
        data["fof_halo_px"].unit
        == data["fof_halo_mass"].unit / data["fof_halo_com_vx"].unit
    )
    assert np.all(
        np.isclose(
            data["fof_halo_px"].value,
            data["fof_halo_mass"].value / data["fof_halo_com_vx"].value,
        )
    )


def test_derive_chain(properties_path):
    ds = oc.open(properties_path)
    derived = oc.col("fof_halo_mass") * (
        oc.col("fof_halo_com_vx") * oc.col("fof_halo_com_vy")
    )
    ds = ds.with_new_columns(fof_halo_p_sqr=derived)
    data = ds.get_data()
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


def test_scalars(properties_path):
    ds = oc.open(properties_path)
    derived1 = oc.col("fof_halo_mass") * 5
    derived2 = 3.0 * oc.col("fof_halo_mass")
    derived3 = 1 / oc.col("fof_halo_mass")
    derived4 = oc.col("fof_halo_mass") / 2.0
    ds = ds.with_new_columns(
        derived1=derived1, derived2=derived2, derived3=derived3, derived4=derived4
    )

    data = ds.get_data()
    assert np.all(np.isclose(data["derived1"], data["fof_halo_mass"] * 5))
    assert np.all(np.isclose(data["derived2"], data["fof_halo_mass"] * 3))
    assert np.all(np.isclose(data["derived3"], 1 / data["fof_halo_mass"]))
    assert np.all(np.isclose(data["derived4"], data["fof_halo_mass"] / 2))


def test_norm_(properties_path):
    ds = oc.open(properties_path)

    total_speed = norm_cols("fof_halo_com_vx", "fof_halo_com_vy", "fof_halo_com_vz")

    ke = 0.5 * oc.col("fof_halo_mass") * total_speed**2
    ds = ds.with_new_columns(ke=ke)
    data = ds.get_data()
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


def test_derive_unit_change(properties_path):
    ds = oc.open(properties_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    comoving_data = ds.get_data()
    comoving_unit = comoving_data["fof_halo_px"].unit
    ds = ds.with_units("scalefree")
    scalefree_data = ds.get_data()
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


def test_derive_mask(properties_path):
    ds = oc.open(properties_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    comoving_data = ds.get_data()["fof_halo_px"]
    val = 0.5 * (comoving_data.max() - comoving_data.min()) + comoving_data.min()
    ds_masked = ds.filter(oc.col("fof_halo_px") > val)

    assert all(ds_masked.get_data()["fof_halo_px"] > val)


def test_derive_children(properties_path):
    ds = oc.open(properties_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    derived2 = 0.5 * oc.col("fof_halo_px") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(derived2=derived2)
    data = ds.get_data()
    assert np.all(
        np.isclose(
            data["derived2"], 0.5 * data["fof_halo_mass"] * data["fof_halo_com_vx"] ** 2
        )
    )


def test_derive_children_select(properties_path):
    ds = oc.open(properties_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    derived2 = 0.5 * oc.col("fof_halo_px") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(derived2=derived2)
    derived_data = ds.get_data()["derived2"]

    to_select = ["fof_halo_com_vy", "derived2"]
    ds = ds.select(to_select)
    data = ds.get_data()
    assert set(data.columns) == set(to_select)
    assert np.all(derived_data == data["derived2"])


def test_derive_structure_collection(properties_path, particles_path):
    ds = oc.open(properties_path, particles_path)
    ds = ds.with_new_columns("dm_particles", gpe=oc.col("mass") * oc.col("phi"))
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14)
    ds = ds.take(1, at="random")
    for halo in ds.objects(["dm_particles"]):
        particles = halo["dm_particles"]
        assert "gpe" in particles.columns


def test_derive_invalid_units(properties_path):
    ds = oc.open(properties_path)
    invalid_col = oc.col("fof_halo_com_vx") ** 2 + oc.col("fof_halo_com_vy")

    with pytest.raises(UnitsError):
        ds = ds.with_new_columns(invalid_col=invalid_col)


def test_derive_divide_by_quantity(properties_path):
    ds = oc.open(properties_path)
    unitless_radius = oc.col("sod_halo_radius") / (1 * u.Mpc)
    data = ds.select("sod_halo_radius", unitless_radius=unitless_radius).get_data()
    assert np.all(data["sod_halo_radius"].value == data["unitless_radius"])


def test_derive_chain_divide_by_quantity(properties_path):
    ds = oc.open(properties_path)
    inertia = oc.col("sod_halo_mass") * oc.col("sod_halo_radius") ** 2
    massless_inertia = inertia / (1 * u.Msun)

    data = ds.select(inertia=inertia, massless_inertia=massless_inertia).get_data()
    assert np.all(data["inertia"].value == data["massless_inertia"].value)
    assert data["inertia"].unit == u.Msun * u.Mpc**2
    assert data["massless_inertia"].unit == u.Mpc**2


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_derived_symbolic_conversion(properties_path):
    ds = oc.open(properties_path)
    ds = ds.with_new_columns(
        fof_halo_com_px=oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    )
    ds_converted = ds.with_units(fof_halo_com_px=u.kg * u.lyr / u.yr)
    original_data = ds.select("fof_halo_com_px").get_data()
    converted_data = ds_converted.select("fof_halo_com_px").get_data()
    assert np.all(original_data.to(u.kg * u.lyr / u.yr) == converted_data)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_derived_log(properties_path):
    ds = oc.open(properties_path)
    ds = ds.with_new_columns(fof_halo_mass_dex=oc.col("fof_halo_mass").log10())
    data = ds.select(("fof_halo_mass", "fof_halo_mass_dex")).get_data()
    assert np.all(
        np.log10(data["fof_halo_mass"].value) == data["fof_halo_mass_dex"].value
    )
    assert u.DexUnit(data["fof_halo_mass"].unit) == data["fof_halo_mass_dex"].unit


def test_derive_zscore(properties_path):
    ds = oc.open(properties_path)
    m = oc.col("fof_halo_mass")
    ds = ds.with_new_columns(zscore=(m - m.mean()) / m.std())
    data = ds.select(("fof_halo_mass", "zscore")).get_data()
    expected = (
        data["fof_halo_mass"].value - np.mean(data["fof_halo_mass"].value)
    ) / np.std(data["fof_halo_mass"].value)
    assert np.all(np.isclose(data["zscore"].value, expected))
    assert data["zscore"].unit == u.dimensionless_unscaled
    assert np.isclose(np.mean(data["zscore"].value), 0.0, atol=1e-6)
    assert np.isclose(np.std(data["zscore"].value), 1.0)


def test_derive_min_max_scaling(properties_path):
    ds = oc.open(properties_path)
    m = oc.col("fof_halo_mass")
    ds = ds.with_new_columns(scaled=(m - m.min()) / (m.max() - m.min()))
    scaled = ds.select("scaled").get_data()
    assert np.isclose(scaled.min().value, 0.0)
    assert np.isclose(scaled.max().value, 1.0)
    assert scaled.unit == u.dimensionless_unscaled


def test_derive_iqr_robust_scaling(properties_path):
    ds = oc.open(properties_path)
    m = oc.col("fof_halo_mass")
    iqr = m.quantile(0.75) - m.quantile(0.25)
    ds = ds.with_new_columns(robust=(m - m.median()) / iqr)
    data = ds.select(("fof_halo_mass", "robust")).get_data()
    raw = data["fof_halo_mass"].value
    expected = (raw - np.median(raw)) / (
        np.quantile(raw, 0.75) - np.quantile(raw, 0.25)
    )
    assert np.all(np.isclose(data["robust"].value, expected))
    assert np.isclose(np.median(data["robust"].value), 0.0)


def test_derive_sum_normalization(properties_path):
    ds = oc.open(properties_path)
    m = oc.col("fof_halo_mass")
    ds = ds.with_new_columns(mass_fraction=m / m.sum())
    data = ds.select(("fof_halo_mass", "mass_fraction")).get_data()
    assert np.isclose(np.sum(data["mass_fraction"].value), 1.0)
    assert data["mass_fraction"].unit == u.dimensionless_unscaled


def test_derive_var_squares_units(properties_path):
    ds = oc.open(properties_path)
    m = oc.col("fof_halo_mass")
    ds = ds.with_new_columns(centered_sq=(m - m.mean()) ** 2 / m.var())
    data = ds.select(("fof_halo_mass", "centered_sq")).get_data()
    raw = data["fof_halo_mass"].value
    expected = (raw - np.mean(raw)) ** 2 / np.var(raw)
    assert np.all(np.isclose(data["centered_sq"].value, expected))
    assert data["centered_sq"].unit == u.dimensionless_unscaled


def test_derive_scalar_preserves_units(properties_path):
    ds = oc.open(properties_path)
    m = oc.col("fof_halo_mass")
    ds = ds.with_new_columns(centered=m - m.mean())
    data = ds.select(("fof_halo_mass", "centered")).get_data()
    assert data["centered"].unit == data["fof_halo_mass"].unit
    expected = data["fof_halo_mass"].value - np.mean(data["fof_halo_mass"].value)
    assert np.all(np.isclose(data["centered"].value, expected))


def test_derive_scalar_on_derived_column(properties_path):
    ds = oc.open(properties_path)
    px = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(px_norm=(px - px.mean()) / px.std())
    data = ds.select(("fof_halo_mass", "fof_halo_com_vx", "px_norm")).get_data()
    raw = data["fof_halo_mass"].value * data["fof_halo_com_vx"].value
    expected = (raw - np.mean(raw)) / np.std(raw)
    assert np.all(np.isclose(data["px_norm"].value, expected))
    assert data["px_norm"].unit == u.dimensionless_unscaled


def test_derive_scalar_reflects_prior_filter(properties_path):
    """A reduction runs over the rows materialized at evaluation time, so a filter
    applied before the with_new_columns call changes the resulting scalar."""
    ds = oc.open(properties_path)
    m = oc.col("fof_halo_mass")
    full_mean = float(np.mean(ds.select("fof_halo_mass").get_data().value))

    filtered = ds.filter(m > 1e13)
    raw_filtered = filtered.select("fof_halo_mass").get_data().value
    expected_mean = float(np.mean(raw_filtered))
    assert not np.isclose(full_mean, expected_mean)

    filtered = filtered.with_new_columns(zscore=(m - m.mean()) / m.std())
    data = filtered.select(("fof_halo_mass", "zscore")).get_data()
    expected = (raw_filtered - expected_mean) / np.std(raw_filtered)
    assert np.all(np.isclose(data["zscore"].value, expected))


def test_derive_scalar_arithmetic_with_quantity(properties_path):
    ds = oc.open(properties_path)
    m = oc.col("fof_halo_mass")
    ds = ds.with_new_columns(shifted=m - (m.mean() + 1e10 * u.Msun))
    data = ds.select(("fof_halo_mass", "shifted")).get_data()
    raw = data["fof_halo_mass"]
    expected = raw - (np.mean(raw) + 1e10 * u.Msun)
    assert np.all(np.isclose(data["shifted"].value, expected.value))
    assert data["shifted"].unit == raw.unit
