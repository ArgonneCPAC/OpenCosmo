"""
Characterization tests for the user-facing semantics of HealpixMap.

These pin pixel/row alignment under row operations, derived metadata, column
handling, and output formats that must survive the move to a HealpixMapState.
"""

import astropy.units as u
import healpy as hp
import healsparse as hsp
import numpy as np
import opencosmo as oc
import pytest
from opencosmo.spatial.region import HealpixRegion

CENTER = (45 * u.deg, -45 * u.deg)


@pytest.fixture
def full_map(test_data):
    return oc.open(test_data.healpix_map)


@pytest.fixture
def small_map(full_map):
    return full_map.bound(oc.make_cone(CENTER, 2 * u.deg))


@pytest.fixture
def lightcone_with_map(test_data):
    return oc.open(
        test_data.lightcone.step(600).halo_properties,
        test_data.lightcone.step(601).halo_properties,
        test_data.healpix_map,
    )


def _values_by_pixel(healpix_map):
    pixels, data = healpix_map.get_data("raw")
    return pixels, data


def _assert_aligned(derived, base):
    base_pixels, base_data = _values_by_pixel(base)
    pixels, data = _values_by_pixel(derived)

    assert len(derived) == len(derived.pixels) == len(pixels)
    assert np.array_equal(np.sort(derived.pixels), pixels)
    assert isinstance(derived.region, HealpixRegion)
    assert np.array_equal(np.sort(derived.region.pixels), pixels)

    rows = np.searchsorted(base_pixels, pixels)
    assert np.array_equal(base_pixels[rows], pixels)
    for name, values in data.items():
        assert np.array_equal(values, base_data[name][rows])


ROW_OPERATIONS = {
    "filter": lambda m: m.filter(oc.col("tsz") < 0),
    "take_start": lambda m: m.take(100, at="start"),
    "take_end": lambda m: m.take(100, at="end"),
    "take_random": lambda m: m.take(100, at="random"),
    "take_range": lambda m: m.take_range(200, 700),
    "take_rows": lambda m: m.take_rows(np.array([3, 1, 50, 900, 2000])),
    "bound_cone": lambda m: m.bound(oc.make_cone(CENTER, 1 * u.deg)),
    "bound_skybox": lambda m: m.bound(oc.make_skybox((44, -46), (46, -44))),
    "sorted_take_start": lambda m: m.sort_by("tsz").take(100, at="start"),
    "sorted_take_end": lambda m: m.sort_by("tsz", invert=True).take(100, at="end"),
    "sorted_take_range": lambda m: m.sort_by("tsz").take_range(10, 60),
    "sorted_take_rows": lambda m: m.sort_by("ksz").take_rows(np.array([0, 5, 9])),
    "chained": lambda m: m.take_range(100, 5000).filter(oc.col("tsz") < 0).take(20),
}


@pytest.mark.parametrize("operation", ROW_OPERATIONS.values(), ids=ROW_OPERATIONS)
def test_row_operations_keep_pixels_aligned(small_map, operation):
    _assert_aligned(operation(small_map), small_map)


def test_sorted_take_selects_extreme_values(small_map):
    _, data = small_map.get_data("raw")
    tsz = np.sort(data["tsz"])

    smallest = small_map.sort_by("tsz").take(10, at="start")
    largest = small_map.sort_by("tsz", invert=True).take(10, at="start")

    assert np.array_equal(np.sort(smallest.get_data("raw")[1]["tsz"]), tsz[:10])
    assert np.array_equal(np.sort(largest.get_data("raw")[1]["tsz"]), tsz[-10:])


def test_sorted_take_range_matches_sorted_slice(small_map):
    _, data = small_map.get_data("raw")
    tsz = np.sort(data["tsz"])

    taken = small_map.sort_by("tsz").take_range(20, 40)
    assert np.array_equal(np.sort(taken.get_data("raw")[1]["tsz"]), tsz[20:40])


def test_take_errors(small_map):
    with pytest.raises(ValueError):
        small_map.take(len(small_map) + 1)
    with pytest.raises(ValueError):
        small_map.take_range(0, len(small_map) + 1)
    with pytest.raises(ValueError):
        small_map.take_rows(np.array([len(small_map)]))
    with pytest.raises(ValueError):
        small_map.take(10, at="middle")


def test_opened_map_metadata(full_map):
    assert full_map.nside == 2048
    assert full_map.nside_lr == 64
    assert full_map.ordering == "NESTED"
    assert full_map.z_range == (0.0, 5.0)
    assert full_map.full_sky
    assert len(full_map) == hp.nside2npix(2048)
    assert np.array_equal(full_map.pixels[:5], np.arange(5))
    assert full_map.columns == ["ksz", "tsz"]
    assert set(full_map.descriptions) == {"ksz", "tsz"}


@pytest.mark.parametrize(
    "operation",
    [
        lambda m: m.bound(oc.make_cone(CENTER, 2 * u.deg)),
        lambda m: m.take_rows(np.array([0, 1, 2])),
        lambda m: m.take(10, at="start"),
        lambda m: m.take_range(0, 10),
    ],
    ids=["bound", "take_rows", "take", "take_range"],
)
def test_row_operations_clear_full_sky_and_keep_metadata(full_map, operation):
    restricted = operation(full_map)
    assert not restricted.full_sky
    assert restricted.nside == full_map.nside
    assert restricted.nside_lr == full_map.nside_lr
    assert restricted.ordering == full_map.ordering
    assert restricted.z_range == full_map.z_range
    assert restricted.columns == full_map.columns


def test_filter_clears_full_sky(small_map):
    assert not small_map.filter(oc.col("tsz") < 0).full_sky


@pytest.mark.parametrize(
    "operation",
    [
        lambda m: m.select("tsz"),
        lambda m: m.sort_by("tsz"),
        lambda m: m.with_new_columns(both=oc.col("tsz") + oc.col("ksz")),
    ],
    ids=["select", "sort_by", "with_new_columns"],
)
def test_column_operations_keep_full_sky(full_map, operation):
    assert operation(full_map).full_sky


def test_with_resolution_full_sky(full_map):
    downgraded = full_map.with_resolution(64)
    assert downgraded.full_sky
    assert downgraded.nside == 64
    assert downgraded.nside_lr == full_map.nside_lr
    assert downgraded.z_range == full_map.z_range
    assert len(downgraded) == hp.nside2npix(64)
    assert np.array_equal(downgraded.pixels, np.arange(hp.nside2npix(64)))


def test_with_resolution_partial(small_map):
    downgraded = small_map.with_resolution(small_map.nside // 4)
    expected = np.unique(small_map.pixels // 16)

    assert not downgraded.full_sky
    assert downgraded.nside == small_map.nside // 4
    assert np.array_equal(downgraded.pixels, expected)
    assert downgraded.columns == small_map.columns
    assert downgraded.descriptions == small_map.descriptions

    _, data = small_map.get_data("raw")
    pixels, low_data = downgraded.get_data("raw")
    assert np.array_equal(pixels, expected)
    parents = small_map.get_data("raw")[0] // 16
    for name, values in low_data.items():
        means = np.array([data[name][parents == p].mean() for p in pixels[:20]])
        assert np.allclose(values[:20], means)


def test_with_resolution_rejects_higher_nside(small_map):
    same = small_map.with_resolution(small_map.nside)
    assert same.nside == small_map.nside
    assert np.array_equal(same.pixels, small_map.pixels)
    with pytest.raises(ValueError):
        small_map.with_resolution(small_map.nside * 2)


def test_with_resolution_then_take(small_map):
    downgraded = small_map.with_resolution(small_map.nside // 2)
    _assert_aligned(downgraded.take_range(10, 50), downgraded)
    _assert_aligned(downgraded.take_rows(np.array([0, 7, 30])), downgraded)


def test_with_resolution_then_filter(small_map):
    downgraded = small_map.with_resolution(small_map.nside // 2)
    _assert_aligned(downgraded.filter(oc.col("tsz") < 0), downgraded)


def test_select_and_drop_columns(small_map):
    assert small_map.select("tsz").columns == ["tsz"]
    assert small_map.drop("tsz").columns == ["ksz"]
    with pytest.raises(ValueError):
        small_map.drop("not_a_column")
    with pytest.raises(ValueError):
        small_map.select("not_a_column")
    assert set(small_map.select("tsz").descriptions) == {"tsz"}


def test_sort_by_unknown_column(small_map):
    with pytest.raises(ValueError):
        small_map.sort_by("not_a_column")


def test_sorted_select_keeps_sort(small_map):
    selected = small_map.sort_by("tsz").select("ksz")
    _, data = small_map.get_data("raw")
    smallest = selected.take(10, at="start")
    expected_pixels = np.sort(
        small_map.get_data("raw")[0][np.argsort(data["tsz"])[:10]]
    )
    assert np.array_equal(smallest.get_data("raw")[0], expected_pixels)


def test_sorted_select_hides_sort_column(small_map):
    selected = small_map.sort_by("tsz").select("ksz")
    assert selected.columns == ["ksz"]
    assert isinstance(selected.get_data("healsparse"), hsp.HealSparseMap)


def test_get_data_single_column(small_map):
    single = small_map.select("tsz")
    assert isinstance(single.get_data(), hsp.HealSparseMap)
    assert isinstance(single.get_data("healsparse"), hsp.HealSparseMap)
    assert isinstance(single.data, hsp.HealSparseMap)

    healpix = single.get_data("healpix")
    assert isinstance(healpix, np.ma.MaskedArray)
    assert len(healpix) == hp.nside2npix(small_map.nside)

    pixels, data = single.get_data("raw")
    assert list(data) == ["tsz"]
    assert np.array_equal(pixels, np.sort(small_map.pixels))


def test_get_data_several_columns(small_map):
    healsparse = small_map.get_data("healsparse")
    assert set(healsparse) == {"ksz", "tsz"}
    assert all(isinstance(m, hsp.HealSparseMap) for m in healsparse.values())

    healpix = small_map.get_data("healpix")
    assert set(healpix) == {"ksz", "tsz"}
    assert all(isinstance(m, np.ma.MaskedArray) for m in healpix.values())

    pixels, raw = small_map.get_data("raw")
    assert set(raw) == {"ksz", "tsz"}
    for name, values in raw.items():
        assert np.array_equal(healsparse[name].get_values_pix(pixels), values)
        assert np.array_equal(healpix[name][pixels], values)


def test_get_data_healpix_masks_uncovered_pixels(small_map):
    pixels, raw = small_map.get_data("raw")
    healpix = small_map.get_data("healpix")
    npix = hp.nside2npix(small_map.nside)

    for name, values in raw.items():
        masked = healpix[name]
        assert np.array_equal(np.nonzero(~masked.mask)[0], pixels)
        assert masked.count() == len(pixels)
        assert np.array_equal(masked.compressed(), values)
        assert np.all(masked.data[masked.mask] == hp.UNSEEN)
        assert np.array_equal(masked.filled(), masked.data)
        assert np.all(hp.ma(masked.filled()).mask == masked.mask)
        assert len(masked) == npix

    # Each column has its own mask.
    healpix["tsz"].mask[0] = not healpix["tsz"].mask[0]
    assert healpix["ksz"].mask[0] != healpix["tsz"].mask[0]


def test_get_data_full_sky_healpix_is_unmasked(full_map):
    healpix = full_map.select("tsz").get_data("healpix")
    assert not isinstance(healpix, np.ma.MaskedArray)
    assert len(healpix) == hp.nside2npix(full_map.nside)


def test_get_data_arguments(small_map):
    with pytest.raises(ValueError):
        small_map.get_data("numpy")
    with pytest.warns(UserWarning):
        data = small_map.select("tsz").get_data(output="healpix")
    assert isinstance(data, np.ma.MaskedArray)

    low = small_map.get_data("raw", nside_out=small_map.nside // 2)
    assert np.array_equal(
        low[0], small_map.with_resolution(small_map.nside // 2).get_data("raw")[0]
    )


def test_evaluate(small_map):
    def offset(tsz, ksz):
        return tsz - ksz

    result = small_map.evaluate(offset, vectorize=True, insert=False)
    assert set(result) == {"offset"}
    assert len(result["offset"]) == len(small_map)

    inserted = small_map.evaluate(offset, vectorize=True, insert=True)
    assert isinstance(inserted, oc.HealpixMap)
    assert inserted.columns == ["ksz", "tsz", "offset"]
    _, data = inserted.get_data("raw")
    assert np.allclose(data["offset"], data["tsz"] - data["ksz"])


def test_evaluate_none_insert_raises(small_map):
    def nothing(tsz):
        return None

    with pytest.raises(ValueError):
        small_map.evaluate(nothing, vectorize=True, insert=True)


def test_evaluate_none_noinsert(small_map):
    def nothing(tsz):
        return None

    assert small_map.evaluate(nothing, vectorize=True, insert=False) == {
        "nothing": None
    }


def test_with_new_columns_array(small_map):
    values = np.arange(len(small_map), dtype=np.float32)
    added = small_map.with_new_columns(index=values)
    _, data = added.get_data("raw")
    assert np.array_equal(data["index"], values)
    with pytest.raises(ValueError):
        small_map.with_new_columns(index=values[:-1])


def test_with_units_not_supported(small_map):
    with pytest.raises(NotImplementedError):
        small_map.with_units("physical")


def test_rows(small_map):
    taken = small_map.take(3, at="start")
    rows = list(taken.rows())
    assert len(rows) == 3
    assert set(rows[0]) == {"ksz", "tsz"}


def test_header_parameter_lookup(small_map):
    assert small_map.cosmology is small_map.header.cosmology or (
        small_map.cosmology == small_map.header.cosmology
    )
    assert "cosmology" in dir(small_map)


def test_repr(small_map):
    text = repr(small_map.take(5, at="start"))
    assert text.startswith("OpenCosmo Healpix Map Dataset (length=5)")
    assert "Cosmology:" in text


@pytest.mark.parametrize(
    "operation",
    [
        lambda m: m,
        lambda m: m.take_range(100, 600),
        lambda m: m.with_resolution(m.nside // 2),
        lambda m: m.sort_by("tsz").select("ksz"),
    ],
    ids=["bound", "take_range", "with_resolution", "sorted_select"],
)
def test_write_round_trip(small_map, operation, tmp_path):
    original = operation(small_map)
    oc.write(tmp_path / "map.hdf5", original)
    written = oc.open(tmp_path / "map.hdf5")

    assert isinstance(written, oc.HealpixMap)
    assert not written.full_sky
    assert written.nside == original.nside
    pixels, data = original.get_data("raw")
    written_pixels, written_data = written.get_data("raw")
    assert np.array_equal(pixels, written_pixels)
    assert set(written_data) == set(data)
    for name, values in data.items():
        assert np.array_equal(values, written_data[name])


def test_write_round_trip_pixel_set(small_map, tmp_path):
    original = small_map.take_range(100, 600)
    oc.write(tmp_path / "map.hdf5", original)
    written = oc.open(tmp_path / "map.hdf5")

    assert isinstance(written, oc.HealpixMap)
    assert not written.full_sky
    assert written.columns == original.columns
    assert np.array_equal(np.sort(written.pixels), np.sort(original.pixels))


def test_lightcone_bound_restricts_map(lightcone_with_map):
    region = oc.make_cone(CENTER, 2 * u.deg)
    bounded = lightcone_with_map.bound(region)
    expected = lightcone_with_map.map.bound(region)
    assert isinstance(bounded.map, oc.HealpixMap)
    assert np.array_equal(bounded.map.pixels, expected.pixels)
    assert not bounded.map.full_sky


def test_lightcone_select_routes_map_columns(lightcone_with_map):
    selected = lightcone_with_map.select("ksz", "fof_halo_mass")
    assert selected.columns == ["fof_halo_mass"]
    assert selected.map.columns == ["ksz"]
    assert selected.map_columns == ["ksz"]

    catalog_only = lightcone_with_map.select("fof_halo_mass")
    assert catalog_only.map.columns == ["ksz", "tsz"]


def test_lightcone_row_operations_keep_map(lightcone_with_map):
    taken = lightcone_with_map.take(10)
    assert len(taken.map) == len(lightcone_with_map.map)
    assert taken.map.full_sky
