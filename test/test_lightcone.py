import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.cosmology import units as cu
from numpy import random

import opencosmo as oc


@pytest.fixture
def haloproperties_600_path(lightcone_path):
    return lightcone_path / "step_600" / "haloproperties.hdf5"


@pytest.fixture
def haloproperties_601_path(lightcone_path):
    return lightcone_path / "step_601" / "haloproperties.hdf5"


@pytest.fixture
def all_files():
    return ["haloparticles.hdf5", "haloproperties.hdf5", "haloprofiles.hdf5"]


@pytest.fixture
def structure_600(lightcone_path, all_files):
    return [lightcone_path / "step_600" / f for f in all_files]


@pytest.fixture
def structure_601(lightcone_path, all_files):
    return [lightcone_path / "step_601" / f for f in all_files]


def test_create_theta_phi_coords(haloproperties_600_path, haloproperties_601_path):
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    data = ds.select(("ra", "dec", "theta", "phi")).get_data()
    assert data["ra"].unit == u.deg
    assert data["dec"].unit == u.deg

    ra = (data["phi"] * u.rad).to(u.deg)
    dec = ((np.pi / 2 - data["theta"]) * u.rad).to(u.deg)

    assert np.allclose(data["ra"], ra, atol=0.0001, rtol=1e-2)
    assert np.allclose(data["dec"], dec, atol=0.0001, rtol=1e-2)


def test_lightcone_physical_units(haloproperties_600_path):
    ds_comoving = oc.open(haloproperties_600_path)
    ds_physical = ds_comoving.with_units("physical")
    data_comoving = ds_comoving.get_data()
    data_physical = ds_physical.get_data()
    assert np.all(
        data_physical["fof_halo_com_x"]
        == (data_comoving["fof_halo_com_x"] * data_comoving["fof_halo_center_a"])
    )
    assert np.all(
        data_physical["fof_halo_com_vx"]
        == (data_comoving["fof_halo_com_vx"] * data_comoving["fof_halo_center_a"])
    )


def test_lc_collection_restrict_z(haloproperties_600_path, haloproperties_601_path):
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    original_redshifts = ds.select("redshift").get_data()
    ds = ds.with_redshift_range(0.040, 0.0405)
    redshifts = ds.select("redshift").get_data()
    masked_redshifts = (original_redshifts > 0.04) & (original_redshifts < 0.0405)
    assert np.all((redshifts > 0.04) & (redshifts < 0.0405))
    assert np.sum(masked_redshifts) == len(redshifts)


def test_lc_collection_pixel_search(haloproperties_600_path, haloproperties_601_path):
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    pixels = ds.get_pixels(64)

    all_coordinates = ds.select("theta", "phi").get_data("numpy")
    all_pixels = hp.ang2pix(
        64, all_coordinates["theta"], all_coordinates["phi"], nest=True
    )

    assert np.all(np.unique(all_pixels) == pixels)

    pixels_to_search = np.sort(np.random.choice(pixels, 20, replace=False))
    ds_bound = ds.pixel_search(pixels_to_search)

    bound_coordinates = ds_bound.select("ra", "dec").get_data("numpy")
    bound_pixels = np.unique(
        hp.ang2pix(
            64,
            bound_coordinates["ra"],
            bound_coordinates["dec"],
            lonlat=True,
            nest=True,
        )
    )
    assert np.all(pixels_to_search == bound_pixels)
    expected_index = np.isin(all_pixels, pixels_to_search)
    found_halo_tags = ds_bound.select("fof_halo_tag").get_data()
    expected_halo_tags = ds.select("fof_halo_tag").get_data()[expected_index]
    assert np.all(found_halo_tags == expected_halo_tags)


def test_lc_collection_write(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    ds = ds.with_redshift_range(0.040, 0.0405)
    original_length = len(ds)
    oc.write(tmp_path / "lightcone.hdf5", ds)
    ds = oc.open(tmp_path / "lightcone.hdf5")
    data = ds.select("redshift").get_data()
    assert data.min() >= 0.04 and data.max() <= 0.0405
    assert len(data) == original_length
    assert ds.z_range == (0.04, 0.0405)


def test_lc_collection_select(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    columns = ds.columns
    to_select = set(random.choice(columns, 10))

    ds = ds.select(to_select)
    columns_found = set(ds.get_data().columns)

    assert columns_found == to_select


def test_lc_collection_select_complex(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    columns = ds.columns
    to_select = set(random.choice(columns, 10))
    to_select_2 = set(random.choice(columns, 5))

    ds = ds.select(*list(to_select), to_select_2)
    columns_found = set(ds.get_data().columns)

    assert columns_found == to_select.union(to_select_2)


def test_lc_collection_select_numpy(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    columns = ds.columns
    to_select = set(random.choice(columns, 10))

    ds = ds.select(to_select)
    data = ds.get_data("numpy")
    assert isinstance(data, dict)
    assert all(ts in data.keys() for ts in to_select)
    assert all(isinstance(col, np.ndarray) for col in data.values())


def test_lc_collection_drop_complex(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    columns = ds.columns
    to_drop = set(random.choice(columns, 10))
    to_drop_2 = set(random.choice(columns, 10))

    ds = ds.drop(*to_drop, list(to_drop_2))
    columns_found = set(ds.get_data().columns)

    assert not columns_found.intersection(to_drop.union(to_drop_2))


def test_lc_collection_drop(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    columns = ds.columns
    to_drop = set(random.choice(columns, 10))

    ds = ds.drop(to_drop)
    columns_found = set(ds.get_data().columns)

    assert not columns_found.intersection(to_drop)


def test_lc_collection_take(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    n_to_take = int(0.75 * len(ds))
    ds_start = ds.take(n_to_take, "start")
    ds_end = ds.take(n_to_take, "end")
    ds_random = ds.take(n_to_take, "random")
    tags = ds.select("fof_halo_tag").get_data()
    tags_start = ds_start.select("fof_halo_tag").get_data()
    tags_end = ds_end.select("fof_halo_tag").get_data()
    tags_random = ds_random.select("fof_halo_tag").get_data()
    assert np.all(tags[:n_to_take] == tags_start)
    assert np.all(tags[-n_to_take:] == tags_end)
    assert len(tags_random) == n_to_take and len(set(tags_random)) == len(tags_random)


def test_lc_collection_range(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    start = int(0.25 * len(ds))
    end = int(0.75 * len(ds))

    ds_range = ds.take_range(start, end)
    halo_tags = ds.select("fof_halo_tag").get_data("numpy")[start:end]
    range_halo_tags = ds_range.select("fof_halo_tag").get_data("numpy")
    assert np.all(halo_tags == range_halo_tags)


def test_lc_collection_take_rows(haloproperties_600_path, haloproperties_601_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    n_to_take = int(0.25 * len(ds))
    rows = np.random.choice(len(ds), n_to_take, replace=False)
    rows.sort()
    ds_rows = ds.take_rows(rows)

    tags = ds.select("fof_halo_tag").get_data("numpy")
    taken_tags = ds_rows.select("fof_halo_tag").get_data("numpy")
    assert np.all(tags[rows] == taken_tags)

    sorted_ds = ds.sort_by("fof_halo_mass").take_rows(rows)
    sorted_tags = sorted_ds.select(("fof_halo_mass", "fof_halo_tag")).get_data()

    data = ds.select(("fof_halo_mass", "fof_halo_tag")).get_data()
    sorted_index = np.argsort(data["fof_halo_mass"])

    assert np.all(
        data["fof_halo_mass"][sorted_index][rows] == sorted_tags["fof_halo_mass"]
    )
    # Verify each returned (tag, mass) pair is internally consistent with the original
    # dataset. We do not compare against a reference sort because sort stability
    # determines which specific halo is selected among ties, and we don't require a
    # particular choice — only that the returned tag actually belongs to a halo with
    # the returned mass.
    tag_order = np.argsort(data["fof_halo_tag"])
    all_tags_sorted = data["fof_halo_tag"][tag_order]
    all_mass_by_tag = data["fof_halo_mass"][tag_order]

    positions = np.searchsorted(all_tags_sorted, sorted_tags["fof_halo_tag"])
    assert np.all(all_mass_by_tag[positions] == sorted_tags["fof_halo_mass"])


def test_lc_collection_derive(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    vsqrd = (
        oc.col("fof_halo_com_vx") ** 2
        + oc.col("fof_halo_com_vy") ** 2
        + oc.col("fof_halo_com_vz") ** 2
    )
    ke = 0.5 * oc.col("fof_halo_mass") * vsqrd
    ds = ds.with_new_columns(ke=ke)
    ke = ds.select("ke")
    assert ke.get_data().unit == u.solMass * u.Unit("km/s") ** 2


def test_lc_collection_add(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    data = np.random.randint(0, 100, len(ds)) * u.deg
    ds = ds.with_new_columns(random=data)
    stored_data = ds.select("random").get_data()
    assert np.all(stored_data == data)


def test_lc_collection_add_with_description(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    data = np.random.randint(0, 100, len(ds)) * u.deg
    fof_px = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    descriptions = {"random": "random data", "px": "com x momentump"}
    ds = ds.with_new_columns(random=data, px=fof_px, descriptions=descriptions)
    descs = ds.descriptions
    for key, value in descriptions.items():
        assert descs[key] == value


def test_lc_get_units(haloproperties_600_path, haloproperties_601_path, tmp_path):
    column_conversions = {"fof_halo_center_x": u.lyr, "ra": u.radian}
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    pre_conversion_units = ds.units
    ds = ds.with_units(None, {u.solMass: u.kg}, **column_conversions)
    post_conversion_units = ds.units

    for name, unit in pre_conversion_units.items():
        if unit == u.solMass:
            assert post_conversion_units[name] == u.kg
        elif name in column_conversions:
            assert unit != post_conversion_units[name]
            assert post_conversion_units[name] == column_conversions[name]
        else:
            assert post_conversion_units[name] == unit


def test_lc_collection_filter(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14)
    assert np.all(ds.get_data()["fof_halo_mass"].value > 1e14)


def test_lc_collection_evaluate(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path).take(200)

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
    ):
        dx = fof_halo_com_x - fof_halo_center_x
        dy = fof_halo_com_y - fof_halo_center_y
        dz = fof_halo_com_z - fof_halo_center_z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    ds_vec = ds.evaluate(offset, vectorize=True, insert=True)
    ds_iter = ds.evaluate(offset, insert=True)

    offset_vec = ds_vec.select("offset").get_data("numpy")
    offset_iter = ds_iter.select("offset").get_data("numpy")
    assert np.all(offset_vec == offset_iter)


def test_lc_collection_mapped_kwargs(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path).take(200)
    mapped_kwargs = {name: np.random.randint(0, 100) for name in ds.keys()}

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        mapped_kwarg,
    ):
        return np.full_like(fof_halo_com_x, mapped_kwarg)

    result = ds.evaluate(
        offset, vectorize=True, insert=True, format="numpy", mapped_kwarg=mapped_kwargs
    )
    for name, ds in result.items():
        assert np.all(
            result[name].select("offset").get_data("numpy") == mapped_kwargs[name]
        )


def test_lc_collection_evaluate_noinsert(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path).take(200)

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
    ):
        dx = fof_halo_com_x - fof_halo_center_x
        dy = fof_halo_com_y - fof_halo_center_y
        dz = fof_halo_com_z - fof_halo_center_z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    result = ds.evaluate(offset, vectorize=True, insert=False)

    assert len(result["offset"]) == len(ds)
    assert np.all(result["offset"] > 0)


class Counter:
    def __init__(self):
        self.__counts = []

    def append_count(self, count: int):
        self.__counts.append(count)

    def get_max_count(self):
        return max(self.__counts)

    @property
    def counts(self):
        return self.__counts


def test_lc_collection_batched(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    batch_size = 1000

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
        counter: Counter,
    ):
        counter.append_count(len(fof_halo_center_x))
        dx = fof_halo_com_x - fof_halo_center_x
        dy = fof_halo_com_y - fof_halo_center_y
        dz = fof_halo_com_z - fof_halo_center_z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    counter = Counter()
    offset = ds.evaluate(
        offset, vectorize=True, insert=False, batch_size=batch_size, counter=counter
    )["offset"]

    assert max(counter.counts) <= batch_size
    assert len(counter.counts) >= len(ds) // batch_size
    assert np.all(offset > 0)
    assert len(offset) == len(ds)


def test_lc_collection_batched_lazy(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    batch_size = 1000

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
        counter: Counter,
    ):
        counter.append_count(len(fof_halo_center_x))
        dx = fof_halo_com_x - fof_halo_center_x
        dy = fof_halo_com_y - fof_halo_center_y
        dz = fof_halo_com_z - fof_halo_center_z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    counter = Counter()
    ds = ds.evaluate(
        offset, vectorize=True, insert=True, batch_size=batch_size, counter=counter
    )

    offset = ds.select("offset").get_data()
    assert max(counter.counts) <= batch_size
    assert len(counter.counts) >= len(ds) // batch_size
    assert np.all(offset > 0)
    assert len(offset) == len(ds)


def test_lc_collection_units(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds_comoving = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds_scalefree = ds_comoving.with_units("scalefree")
    data_scalefree = ds_scalefree.get_data()
    data_comoving = ds_comoving.get_data()
    h = ds_comoving.cosmology.h

    location_columns = [f"fof_halo_com_{dim}" for dim in ("x", "y", "z")]
    for column in location_columns:
        assert data_scalefree[column].unit == u.Mpc / cu.littleh
        assert data_comoving[column].unit == u.Mpc
        assert np.all(
            np.isclose(data_scalefree[column].value, data_comoving[column].value * h)
        )


def test_lc_collection_units_convert(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    location_columns = [f"fof_halo_com_{dim}" for dim in ("x", "y", "z")]
    conversions = {c: u.lyr for c in location_columns}
    ds_converted = ds.with_units(**conversions)
    data_original = ds.select(location_columns).get_data()
    data_converted = ds_converted.select(location_columns).get_data()
    for column in location_columns:
        assert np.all(data_original[column].to(u.lyr) == data_converted[column])


def test_lc_collection_sort(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds = ds.sort_by("fof_halo_mass")
    data = ds.select("fof_halo_mass").get_data()
    assert np.all(data[1:] >= data[:-1])


def test_lc_collection_sort_and_take(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    halo_masses = ds.select("fof_halo_mass").get_data()
    halo_masses = np.sort(halo_masses)
    ds = ds.sort_by("fof_halo_mass")
    ds = ds.take(100, at="start")
    data = ds.select("fof_halo_mass").get_data()
    assert np.all(data[1:] >= data[:-1])
    assert np.all(halo_masses[:100] == data)


def test_write_single_lightcone(haloproperties_600_path, tmp_path):
    ds = oc.open(haloproperties_600_path)
    assert isinstance(ds, oc.Lightcone)
    oc.write(tmp_path / "temp.hdf5", ds)
    ds_written = oc.open(tmp_path / "temp.hdf5")
    assert isinstance(ds, oc.Lightcone)
    assert len(ds) == len(ds_written)


def test_insert_to_sorted(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds = ds.sort_by("fof_halo_mass").take(10_000, at="end")
    random_value = np.random.normal(1.0, 0.1, len(ds))
    ds = ds.with_new_columns(
        random_value=random_value,
        random_mass=oc.col("fof_halo_mass") * oc.col("random_value"),
    )
    data = ds.select(("random_value", "random_mass", "fof_halo_mass")).get_data()
    assert np.all(
        np.isclose(data["random_mass"], data["random_value"] * data["fof_halo_mass"])
    )
    output_path = tmp_path / "data.hdf5"
    oc.write(output_path, ds)
    ds = oc.open(output_path).sort_by("fof_halo_mass")
    data = ds.select(("random_value", "random_mass", "fof_halo_mass")).get_data()
    assert np.all(
        np.isclose(data["random_mass"], data["random_value"] * data["fof_halo_mass"])
    )


def test_lightcone_stacking(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds = ds.take(120_000, at="random")
    for dataset in ds.values():
        assert dataset.header.lightcone["z_range"] != ds.z_range

    fof_tags = ds.select("fof_halo_tag").get_data()
    output_path = tmp_path / "data.hdf5"
    oc.write(output_path, ds)
    ds_new = oc.open(output_path)
    fof_tags_new = ds_new.select("fof_halo_tag").get_data()
    assert len(ds_new.keys()) == 1
    assert np.all(np.unique(fof_tags) == np.unique(fof_tags_new))
    assert ds_new.z_range == ds.z_range
    assert next(iter(ds_new.values())).header.lightcone["z_range"] == ds_new.z_range


def test_lightcone_stacking_nostack(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)

    fof_tags = ds.select("fof_halo_tag").get_data()
    output_path = tmp_path / "data.hdf5"
    oc.write(output_path, ds, _min_size=100)
    ds_new = oc.open(output_path)
    fof_tags_new = ds_new.select("fof_halo_tag").get_data()
    assert np.all(np.unique(fof_tags) == np.unique(fof_tags_new))
    assert ds_new.z_range == ds.z_range


def test_lightcone_structure_collection_open(structure_600):
    c = oc.open(*structure_600)
    assert isinstance(c, oc.StructureCollection)


# ---------------------------------------------------------------------------
# Lightcone-scoped derived scalars
#
# Scalars (col.mean(), col.min(), ...) in expressions registered against a
# Lightcone must reduce across the WHOLE lightcone, not per-child. The
# reduction is lazy: it sees the rows that remain after every prior
# filter/take/with_redshift_range at the moment get_data is called.
# ---------------------------------------------------------------------------


def _global_raw(lc, name):
    """Materialize a single raw child column across the whole lightcone."""
    pieces = [child.select(name).get_data() for child in lc.values()]
    return np.concatenate([np.asarray(p) for p in pieces])


def test_lc_scope_global_zscore(haloproperties_600_path, haloproperties_601_path):
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    m = oc.col("fof_halo_mass")
    derived = lc.with_new_columns(zscore=(m - m.mean()) / m.std())
    data = derived.get_data()

    z = np.asarray(data["zscore"])
    assert np.isclose(z.mean(), 0.0, atol=1e-6)
    assert np.isclose(z.std(), 1.0, atol=1e-6)

    raw = _global_raw(lc, "fof_halo_mass")
    per_child_means = [
        np.mean(np.asarray(child.select("fof_halo_mass").get_data()))
        for child in lc.values()
    ]
    assert not np.isclose(
        per_child_means[0], np.mean(raw), rtol=1e-6
    ) or not np.isclose(per_child_means[1], np.mean(raw), rtol=1e-6), (
        "Per-child means equal the global mean — fixture too uniform to detect the bug"
    )


def test_lc_scope_global_min_max_scaling(
    haloproperties_600_path, haloproperties_601_path
):
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    m = oc.col("fof_halo_mass")
    derived = lc.with_new_columns(scaled=(m - m.min()) / (m.max() - m.min()))
    data = derived.get_data()

    scaled = np.asarray(data["scaled"])
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    assert np.isclose(scaled.min(), 0.0, atol=1e-9)
    assert np.isclose(scaled.max(), 1.0, atol=1e-9)


def test_lc_scope_scalar_select_is_global(
    haloproperties_600_path, haloproperties_601_path
):
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    result = lc.select(min_mass=oc.col("fof_halo_mass").min()).get_data()
    raw = _global_raw(lc, "fof_halo_mass")

    assert isinstance(result, u.Quantity)
    assert result.isscalar or result.shape == ()
    assert np.isclose(float(result.value), float(np.min(raw)))


def test_lc_scope_scalar_select_value_is_global(
    haloproperties_600_path, haloproperties_601_path
):
    """
    Verify the *value* a scope-only scalar selection produces is the global
    reduction across all children, not a per-child one. Guards the
    reducer/scope wiring independently of the return shape.
    """
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    result = lc.select(min_mass=oc.col("fof_halo_mass").min()).get_data()
    raw = _global_raw(lc, "fof_halo_mass")
    arr = np.asarray(result)
    assert np.all(np.isclose(arr, float(np.min(raw))))


def test_lc_scope_with_new_columns_rejects_bare_scalar(
    haloproperties_600_path, haloproperties_601_path
):
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    # Should not raise — was previously a ValueError because the
    # isinstance check at lightcone.py rejected DerivedScalarValue.
    with pytest.raises(
        ValueError, match="Scalar values cannot be added to an existing dataset"
    ):
        _ = lc.with_new_columns(min_mass=oc.col("fof_halo_mass").min())


def test_lc_scope_select_rejects_mixed_scalar_and_column(
    haloproperties_600_path, haloproperties_601_path
):
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    with pytest.raises(ValueError, match="Scalar selections cannot be mixed"):
        lc.select("fof_halo_mass", min_mass=oc.col("fof_halo_mass").min())


def test_lc_scope_filter_against_global_mean(
    haloproperties_600_path, haloproperties_601_path
):
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    m = oc.col("fof_halo_mass")
    raw = _global_raw(lc, "fof_halo_mass")
    threshold = float(np.mean(raw))

    filtered = lc.filter(m > m.mean())
    kept = np.asarray(filtered.select("fof_halo_mass").get_data())

    expected = raw[raw > threshold]
    assert len(kept) == len(expected)
    assert np.isclose(np.sort(kept), np.sort(expected)).all()


def test_lc_scope_lazy_across_redshift_range(
    haloproperties_600_path, haloproperties_601_path
):
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    m = oc.col("fof_halo_mass")

    z_low, z_high = lc.z_range
    z_mid = (z_low + z_high) / 2
    restricted = lc.with_redshift_range(z_low, z_mid)

    raw_full = _global_raw(lc, "fof_halo_mass")
    raw_restricted = _global_raw(restricted, "fof_halo_mass")
    assert not np.isclose(np.mean(raw_full), np.mean(raw_restricted)), (
        "Fixture redshift bins have equal means — can't prove laziness"
    )

    restricted_centered = restricted.with_new_columns(centered=m - m.mean()).get_data()
    centered = np.asarray(restricted_centered["centered"])
    expected = raw_restricted - np.mean(raw_restricted)
    assert np.allclose(np.sort(centered), np.sort(expected), rtol=1e-6)


def test_lc_scope_lazy_across_post_filter(
    haloproperties_600_path, haloproperties_601_path
):
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    m = oc.col("fof_halo_mass")
    raw = _global_raw(lc, "fof_halo_mass")
    threshold = float(np.median(raw))

    derived = lc.with_new_columns(centered=m - m.mean()).filter(m > threshold * u.Msun)
    data = derived.get_data()

    raw_kept = raw[raw > threshold]
    expected_centered = raw_kept - np.mean(raw_kept)
    centered = np.asarray(data["centered"])
    assert len(centered) == len(expected_centered)
    assert np.allclose(np.sort(centered), np.sort(expected_centered), rtol=1e-6)
