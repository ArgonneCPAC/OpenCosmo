import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from opencosmo.collection.healpix_map import state as mapst
from opencosmo.dataset.state import state_from_evaluated_data, state_in_memory
from opencosmo.handler.memory import InMemoryHandler
from opencosmo.spatial.region import HealpixRegion

import opencosmo as oc
from opencosmo.dataset import operations as dsops


@pytest.fixture
def full_map(test_data):
    return oc.open(test_data.healpix_map)


@pytest.fixture
def full_state(full_map):
    return mapst.from_dataset(next(iter(full_map.values()))._state)


@pytest.fixture
def small_state(full_map):
    small_map = full_map.bound(oc.make_cone((45 * u.deg, -45 * u.deg), 2 * u.deg))
    return mapst.from_dataset(next(iter(small_map.values()))._state)


@pytest.fixture
def memory_state(full_state):
    # The worked example from plans/healpix-map-state.md: five rows of a
    # partial map, stored in memory with an explicit pixel per row.
    dataset = state_from_evaluated_data(
        {
            "v": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "w": np.array([4.0, 2.0, 3.0, 0.0, 1.0]),
        },
        full_state.header,
    )
    return mapst.HealpixMapState(dataset, np.array([3, 8, 12, 40, 41]))


def _with_dataset(state, dataset):
    return mapst.HealpixMapState(dataset, state.raw_pixels)


def test_from_dataset_full_sky(full_map, full_state):
    assert full_state.raw_pixels is None
    assert full_state.full_sky
    assert np.array_equal(full_state.pixels, np.arange(hp.nside2npix(full_state.nside)))
    assert full_state.nside == full_map.nside
    assert full_state.nside_lr == full_map.nside_lr
    assert full_state.ordering == full_map.ordering
    assert full_state.z_range == full_map.z_range
    assert full_state.columns == full_map.columns
    assert full_state.descriptions == full_map.descriptions
    assert len(full_state) == len(full_map)


def test_bounded_state_matches_facade(full_map, small_state):
    small_map = full_map.bound(oc.make_cone((45 * u.deg, -45 * u.deg), 2 * u.deg))
    assert not small_state.full_sky
    assert np.array_equal(small_state.pixels, small_map.pixels)
    assert isinstance(small_state.region, HealpixRegion)
    assert np.array_equal(small_state.region.pixels, small_map.pixels)
    assert small_state.region.nside == small_state.nside


def test_explicit_raw_pixels(memory_state):
    assert isinstance(memory_state.dataset.raw_data_handler, InMemoryHandler)
    assert np.array_equal(memory_state.pixels, [3, 8, 12, 40, 41])
    assert not memory_state.full_sky
    assert len(memory_state) == 5


@pytest.mark.parametrize("state_name", ["small_state", "memory_state"])
def test_pixels_follow_take_rows(request, state_name):
    state = request.getfixturevalue(state_name)
    rows = np.array([1, 3])
    taken = _with_dataset(state, dsops.take_rows(state.dataset, rows))
    assert np.array_equal(taken.pixels, state.pixels[rows])


@pytest.mark.parametrize("state_name", ["small_state", "memory_state"])
def test_pixels_follow_filter(request, state_name):
    state = request.getfixturevalue(state_name)
    column = state.columns[0]
    values = dsops.get_data(state.dataset, "numpy")[column]
    mask = values > np.median(values)
    filtered = _with_dataset(
        state, dsops.filter(state.dataset, oc.col(column) > np.median(values))
    )
    assert np.array_equal(filtered.pixels, state.pixels[mask])


@pytest.mark.parametrize("state_name", ["small_state", "memory_state"])
@pytest.mark.parametrize("invert", [False, True])
def test_pixels_follow_sorted_take_rows_in_raw_order(request, state_name, invert):
    state = request.getfixturevalue(state_name)
    column = state.columns[-1]
    values = dsops.get_data(state.dataset, "numpy")[column]
    order = np.argsort(values)
    if invert:
        order = order[::-1]

    rows = np.array([0, 2, 3])
    sorted_dataset = dsops.sort_by(state.dataset, column, invert)
    taken = _with_dataset(state, dsops.take_rows(sorted_dataset, rows))

    assert np.all(np.diff(taken.pixels) > 0)
    assert np.array_equal(taken.pixels, np.sort(state.pixels[order[rows]]))


def test_pixels_ignore_sort_order(memory_state):
    # w = [4, 2, 3, 0, 1], so ascending sort order is rows [3, 4, 1, 2, 0].
    sorted_dataset = dsops.sort_by(memory_state.dataset, "w", False)
    taken = _with_dataset(
        memory_state, dsops.take_rows(sorted_dataset, np.array([0, 2, 3]))
    )
    assert np.array_equal(taken.dataset.raw_index, [3, 1, 2])
    assert np.array_equal(taken.pixels, [8, 12, 40])


def test_worked_example_downgrade_then_filter(memory_state):
    filtered = _with_dataset(
        memory_state, dsops.take_rows(memory_state.dataset, np.array([1, 3]))
    )
    assert np.array_equal(filtered.pixels, [8, 40])

    low_res = filtered.pixels // 4
    downgraded = mapst.HealpixMapState(
        state_from_evaluated_data({"v": np.array([1.0, 3.0])}, filtered.header),
        low_res,
    )
    assert np.array_equal(downgraded.pixels, [2, 10])

    refiltered = _with_dataset(
        downgraded, dsops.filter(downgraded.dataset, oc.col("v") > 2)
    )
    assert np.array_equal(refiltered.pixels, [10])


def test_downgraded_map_state(full_map):
    small_map = full_map.bound(oc.make_cone((45 * u.deg, -45 * u.deg), 2 * u.deg))
    downgraded = small_map.with_resolution(small_map.nside // 2)
    dataset = next(iter(downgraded.values()))._state
    state = mapst.HealpixMapState(dataset, downgraded.pixels)

    assert state.nside == downgraded.nside
    assert np.array_equal(state.pixels, downgraded.pixels)

    filtered = downgraded.filter(oc.col("tsz") < 0)
    filtered_state = _with_dataset(state, next(iter(filtered.values()))._state)
    assert np.array_equal(filtered_state.pixels, filtered.pixels)


def test_full_sky_is_local(full_state):
    assert full_state.full_sky
    taken = _with_dataset(
        full_state, dsops.take_range(full_state.dataset, 0, 10, mode="local")
    )
    assert not taken.full_sky


def test_empty_handler_rejected(full_state):
    dataset = state_in_memory(
        {"v": np.arange(3.0)},
        full_state.header,
        full_state.header.file.unit_convention,
        {},
    )
    with pytest.raises(RuntimeError, match="tracks its rows"):
        mapst.HealpixMapState(dataset, np.array([0, 1, 2]))
