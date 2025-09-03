import astropy.units as u
import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def particle_path(snapshot_path):
    return snapshot_path / "haloparticles.hdf5"


@pytest.fixture
def max_mass(input_path):
    ds = oc.open(input_path)
    sod_mass = ds.data["sod_halo_mass"]
    return 0.95 * sod_mass.max()


def test_open(input_path):
    read_data = oc.open(input_path).data
    with oc.open(input_path) as f:
        open_data = f.data

    assert np.all(read_data == open_data)
    columns = read_data.columns
    assert all(open_data[col].unit == read_data[col].unit for col in columns)


def test_open_close(input_path):
    with oc.open(input_path) as ds:
        file = ds._Dataset__handler._DatasetHandler__file
        assert file["data"] is not None

    with pytest.raises(KeyError):
        file["data"]


def test_dataset_close(input_path):
    ds = oc.open(input_path)
    file = ds._Dataset__handler._DatasetHandler__file
    assert file["data"] is not None
    ds.close()
    with pytest.raises(KeyError):
        file["data"]


def test_filter_oom(input_path, max_mass):
    # Assuming test_open worked, this is the only
    # thing that needs to be directly tested

    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0, oc.col("sod_halo_mass") < max_mass)
        data = ds.data
    assert data["sod_halo_mass"].min() > 0


def test_filter_to_numpy(input_path, max_mass):
    # Assuming test_open worked, this is the only
    # thing that needs to be directly tested

    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0, oc.col("sod_halo_mass") < max_mass)
        data = ds.get_data(output="numpy")
    assert isinstance(data, dict)
    for val in data.values():
        assert isinstance(val, np.ndarray)
    assert data["sod_halo_mass"].min() > 0


def test_take_oom(input_path):
    with oc.open(input_path) as f:
        ds = f.take(10)
        data = ds.data
    assert len(data) == 10


def test_take_sorted(input_path):
    n = 150
    dataset = oc.open(input_path)
    fof_masses = dataset.select("fof_halo_mass").get_data("numpy")
    toolkit_sorted_fof_masses = (
        dataset.order_by("fof_halo_mass")
        .take(150, at="start")
        .select("fof_halo_mass")
        .get_data("numpy")
    )
    manually_sorted_fof_masses = -np.sort(-fof_masses)
    assert np.all(manually_sorted_fof_masses[:n] == toolkit_sorted_fof_masses)
    assert fof_masses.max() == toolkit_sorted_fof_masses[0]


def test_order_by_derived(input_path):
    n = 150
    ds = oc.open(input_path)
    dx = oc.col("fof_halo_com_x") - oc.col("sod_halo_com_x")
    dy = oc.col("fof_halo_com_y") - oc.col("sod_halo_com_y")
    dz = oc.col("fof_halo_com_z") - oc.col("sod_halo_com_z")
    dr = (dx**2 + dy**2 + dz**2) ** (0.5)
    xoff = dr / oc.col("sod_halo_radius")

    ds = ds.with_new_columns(xoff=xoff)
    xoff = ds.select(("xoff", "fof_halo_tag")).get_data("numpy")
    toolkit_sorted_xoff = (
        ds.order_by("xoff")
        .take(n, at="start")
        .select(("fof_halo_tag", "xoff"))
        .get_data("numpy")
    )
    idxs = np.argsort(-xoff["xoff"])
    assert np.all(xoff["xoff"][idxs][:n] == toolkit_sorted_xoff["xoff"])
    assert xoff["xoff"].max() == toolkit_sorted_xoff["xoff"][0]
    assert np.all(xoff["fof_halo_tag"][idxs][:n] == toolkit_sorted_xoff["fof_halo_tag"])


def test_drop(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        dropped_cols = np.random.choice(cols, 10, replace=False)
        selected = ds.drop(dropped_cols)
        selected_data = selected.data

    dropped_cols = set(dropped_cols)
    remaining_cols = set(selected_data.colnames)
    assert not dropped_cols.intersection(remaining_cols)


def test_drop_single(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        dropped_col = np.random.choice(cols)
        remaining = ds.drop(dropped_col)
        remaining_data = remaining.data

    assert dropped_col not in remaining_data.colnames


def test_visit_vectorize_single(input_path):
    ds = oc.open(input_path)

    def fof_total(fof_halo_mass):
        return np.cumsum(fof_halo_mass)

    ds = ds.evaluate(fof_total, vectorize=True, insert=True)
    assert "fof_total" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_total"]).get_data("numpy")
    assert np.all(data["fof_total"] == np.cumsum(data["fof_halo_mass"]))


def test_visit_vectorize_multiple(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    ds = ds.evaluate(fof_px, vectorize=True, insert=True)
    assert "fof_px" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


def test_visit_with_default(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx, random_val=5):
        return fof_halo_mass * fof_halo_com_vx * random_val

    ds = ds.evaluate(fof_px, vectorize=True, insert=True)
    assert "fof_px" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"] * 5)


def test_visit_with_return_none(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx, random_val=5):
        return None

    result = ds.evaluate(fof_px, vectorize=True, insert=True)
    assert result is None


def test_visit_multiple_with_kwargs_numpy(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx, random_value):
        return fof_halo_mass * fof_halo_com_vx + random_value

    random_values = np.random.randint(0, 100, len(ds))
    result = ds.evaluate(
        fof_px, insert=False, format="numpy", vectorize=True, random_value=random_values
    )
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx"]).get_data("numpy")
    assert np.all(
        result["fof_px"]
        == data["fof_halo_mass"] * data["fof_halo_com_vx"] + random_values
    )


def test_visit_multiple_with_iterable_kwargs(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx, random_value):
        return fof_halo_mass * fof_halo_com_vx * random_value

    random_values = np.random.randint(0, 100, len(ds))
    result = ds.evaluate(fof_px, vectorize=False, random_value=random_values)
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx"]).get_data("numpy")
    assert np.all(
        result["fof_px"].value
        == data["fof_halo_mass"] * data["fof_halo_com_vx"] * random_values
    )


def test_visit_vectorize_multiple_noinsert(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    result = ds.evaluate(fof_px, vectorize=True, insert=False)
    data = ds.select(("fof_halo_mass", "fof_halo_com_vx")).data

    assert np.all(result["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


def test_visit_rows_nfw(input_path):
    ds = oc.open(input_path).filter(oc.col("sod_halo_cdelta") > 0)

    def nfw(sod_halo_radius, sod_halo_cdelta, sod_halo_mass):
        r_ = np.logspace(-2, 0, 50)
        A = np.log(1 + sod_halo_cdelta) - sod_halo_cdelta / (1 + sod_halo_cdelta)

        halo_density = sod_halo_mass / (4 / 3 * np.pi * sod_halo_radius**3)
        profile = halo_density / (3 * A * r_) / (1 / sod_halo_cdelta + r_) ** 2
        return {"nfw_radius": r_ * sod_halo_radius, "nfw_profile": profile}

    ds = ds.evaluate(nfw, insert=True)

    assert "nfw_radius" in ds.columns
    assert "nfw_profile" in ds.columns
    profile = ds.select("nfw_profile").get_data("numpy")
    assert profile.shape == (len(ds), 50)


def test_visit_rows_multiple(input_path):
    ds = oc.open(input_path).take(100)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    ds = ds.evaluate(fof_px, vectorize=False, insert=True)
    assert "fof_px" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


def test_visit_rows_single(input_path):
    ds = oc.open(input_path).take(100)

    def fof_random(fof_halo_mass):
        return fof_halo_mass * np.random.randint(0, 100)

    ds = ds.evaluate(fof_random, vectorize=False, insert=True)
    assert "fof_random" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_random"]).get_data()
    assert data["fof_random"].unit == u.solMass
    factor = data["fof_random"] / data["fof_halo_mass"]
    factor = factor.value
    assert np.all(factor == np.floor(factor))


def test_visit_rows_all(input_path):
    ds = oc.open(input_path).take(100)

    def fof_random(halo_properties):
        return np.random.randint(0, 100)

    ds = ds.evaluate(fof_random, vectorize=False, insert=True)
    data = ds.select(["fof_random"]).get_data()
    assert data.dtype == np.int64


def test_visit_rows_all_vectorize(input_path):
    ds = oc.open(input_path).take(100)

    def fof_random(halo_properties):
        return np.random.randint(0, 100, len(halo_properties["fof_halo_tag"]))

    ds = ds.evaluate(fof_random, vectorize=True, insert=True)
    data = ds.select(["fof_random"]).get_data()
    assert data.dtype == np.int64


def test_select_oom(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        selected_cols = np.random.choice(cols, 10, replace=False)
        selected = ds.select(selected_cols)
        selected_data = selected.data

    for col in selected_cols:
        assert np.all(data[col] == selected_data[col])
    assert set(selected_cols) == set(selected_data.columns)


def test_select_single(input_path):
    with oc.open(input_path) as ds:
        for column in ds.columns:
            selected = ds.select(column)
            selected_data = selected.data
            assert isinstance(selected_data, (u.Quantity, np.ndarray))


def test_select_single_numpy(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        selected_cols = np.random.choice(cols)
        selected = ds.select(selected_cols)
        selected_data = selected.get_data("numpy")
    assert isinstance(selected_data, np.ndarray)


def test_write_after_filter(input_path, tmp_path):
    with oc.open(input_path) as ds:
        ds = ds.filter(oc.col("sod_halo_mass") > 0)

        oc.write(tmp_path / "haloproperties.hdf5", ds)

        data = ds.data

    with oc.open(tmp_path / "haloproperties.hdf5") as new_ds:
        filtered_data = new_ds.data
        assert np.all(data == filtered_data)


def test_write_after_derive(input_path, tmp_path):
    with oc.open(input_path) as ds:
        col = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
        ds = ds.with_new_columns(fof_halo_px=col)

        oc.write(tmp_path / "haloproperties.hdf5", ds)

        data = ds.select("fof_halo_px").data

    with oc.open(tmp_path / "haloproperties.hdf5") as new_ds:
        written_data = new_ds.select("fof_halo_px").data
        assert np.all(np.isclose(data, written_data))


def test_write_after_evaluate(input_path, tmp_path):
    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    with oc.open(input_path) as ds:
        ds = ds.evaluate(fof_px, insert=True, vectorize=True)

        oc.write(tmp_path / "haloproperties.hdf5", ds)

        data = ds.select("fof_px").data

    with oc.open(tmp_path / "haloproperties.hdf5") as new_ds:
        written_data = new_ds.select("fof_px").data
        assert np.all(np.isclose(data, written_data))


def test_collect(input_path):
    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0).take(100, at="random").collect()

    assert len(ds.data) == 100


def test_select_collect(input_path):
    with oc.open(input_path) as f:
        ds = (
            f.filter(oc.col("sod_halo_mass") > 0)
            .select(["sod_halo_mass", "fof_halo_mass"])
            .take(100, at="random")
            .collect()
        )

    assert len(ds.data) == 100
    assert set(ds.data.columns) == {"sod_halo_mass", "fof_halo_mass"}
