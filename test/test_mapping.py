from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from uuid import uuid4

import h5py
import numpy as np
import opencosmo.collection.simulation.simulation as simulation_module
import pytest
from opencosmo.mapping.mapping import DatasetMatchSet, rebuild_single_with_new_source

import opencosmo as oc
from opencosmo.index import into_array

if TYPE_CHECKING:
    from pathlib import Path


REFERENCE = "SCIDAC_128_GO"
SIMULATION_A = "KAPPA_2.222_EGW_0.759_SEED_7.810e5_VKIN_5889_EPS_5.257"
SIMULATION_B = "KAPPA_2.984_EGW_0.682_SEED_6e5_VKIN_7286_EPS_4.883"
MAPPED_SIMULATIONS = (REFERENCE, SIMULATION_A, SIMULATION_B)


@pytest.fixture
def mapped_paths(test_data):
    return {
        REFERENCE: test_data.snapshot.mapping_reference,
        SIMULATION_A: test_data.snapshot.scidac(0).halo_properties,
        SIMULATION_B: test_data.snapshot.scidac(1).halo_properties,
    }


def _open_mapped(mapped_paths, mapping_path, simulations=MAPPED_SIMULATIONS):
    return oc.open(*(mapped_paths[name] for name in simulations), mapping_path)


def _expected_pairwise_maps(mapping_path: Path, mapped_paths):
    uuids = {}
    lengths = {}
    for name, path in mapped_paths.items():
        with h5py.File(path) as file:
            uuids[name] = str(file["data"].attrs["uuid"])
            lengths[name] = len(file["data/fof_halo_tag"])

    with h5py.File(mapping_path) as file:
        group = file["map"]
        reference_uuid = str(group.attrs["reference"])
        assert uuids[REFERENCE] == reference_uuid

        primary_a = group[f"primary/{uuids[SIMULATION_A]}/index"][:]
        primary_b = group[f"primary/{uuids[SIMULATION_B]}/index"][:]
        auxiliary = group[f"auxiliary/{uuids[SIMULATION_A]}__{uuids[SIMULATION_B]}"]
        auxiliary_a = auxiliary["source"][:]
        auxiliary_b = auxiliary["target"][:]

    pairwise = {}
    pairwise[(REFERENCE, SIMULATION_A)] = primary_a
    pairwise[(REFERENCE, SIMULATION_B)] = primary_b

    for target, primary in (
        (SIMULATION_A, primary_a),
        (SIMULATION_B, primary_b),
    ):
        inverse = np.full(lengths[target], -1, dtype=np.int64)
        reference_rows = np.flatnonzero(primary >= 0)
        inverse[primary[reference_rows]] = reference_rows
        pairwise[(target, REFERENCE)] = inverse

    a_to_b = np.full(lengths[SIMULATION_A], -1, dtype=np.int64)
    primary_rows = np.flatnonzero((primary_a >= 0) & (primary_b >= 0))
    a_to_b[primary_a[primary_rows]] = primary_b[primary_rows]
    a_to_b[auxiliary_a] = auxiliary_b
    pairwise[(SIMULATION_A, SIMULATION_B)] = a_to_b

    b_to_a = np.full(lengths[SIMULATION_B], -1, dtype=np.int64)
    b_to_a[primary_b[primary_rows]] = primary_a[primary_rows]
    b_to_a[auxiliary_b] = auxiliary_a
    pairwise[(SIMULATION_B, SIMULATION_A)] = b_to_a
    return pairwise


def _assert_matches_mapping(before, matched, source, pairwise):
    source_index = into_array(before[source].index)
    rows_to_keep = np.ones(len(source_index), dtype=bool)

    for target in before.keys() - {source}:
        mapped_rows = pairwise[(source, target)][source_index]
        rows_to_keep &= mapped_rows >= 0
        rows_to_keep &= np.isin(mapped_rows, into_array(before[target].index))

    np.testing.assert_array_equal(
        into_array(matched[source].index), source_index[rows_to_keep]
    )
    for target in before.keys() - {source}:
        np.testing.assert_array_equal(
            into_array(matched[target].index),
            pairwise[(source, target)][source_index[rows_to_keep]],
        )


def _column(dataset, name):
    return np.asarray(dataset.select(name).get_data(format="numpy"))


def _absolute_rows_from_ids(dataset, ids, identifier="fof_halo_tag"):
    dataset_ids = _column(dataset, identifier)
    dataset_index = into_array(dataset.index)
    rows_by_id = dict(zip(dataset_ids.tolist(), dataset_index.tolist(), strict=True))
    return np.asarray([rows_by_id[value] for value in ids.tolist()], dtype=np.int64)


def _expected_source_driven_ids(
    original, expected_source, source, pairwise, identifier="fof_halo_tag"
):
    expected_ids = {source: _column(expected_source, identifier)}
    source_rows = _absolute_rows_from_ids(
        original[source], expected_ids[source], identifier
    )
    for target in original.keys() - {source}:
        target_rows = pairwise[(source, target)][source_rows]
        assert np.all(target_rows >= 0)

        target_ids = _column(original[target], identifier)
        target_index = into_array(original[target].index)
        ids_by_row = dict(zip(target_index.tolist(), target_ids.tolist(), strict=True))
        expected_ids[target] = np.asarray(
            [ids_by_row[row] for row in target_rows.tolist()], dtype=target_ids.dtype
        )
    return expected_ids


def _assert_source_driven_result(
    original, expected_source, actual, source, pairwise, identifier="fof_halo_tag"
):
    """Assert logical row order in every catalog follows ``expected_source``."""
    expected_ids = _expected_source_driven_ids(
        original, expected_source, source, pairwise, identifier
    )
    for name, ids in expected_ids.items():
        np.testing.assert_array_equal(_column(actual[name], identifier), ids)


def _assert_mapping_equal(before, after, identifier="fof_halo_tag"):
    """Assert two collections describe the same matches using stable row IDs."""
    assert set(before) == set(after)
    names = tuple(sorted(before))

    for source in names:
        before_matched = before.match(source)
        after_matched = after.match(source)

        before_ids = [
            np.asarray(before_matched[name].select(identifier).get_data(format="numpy"))
            for name in names
        ]
        after_ids = [
            np.asarray(after_matched[name].select(identifier).get_data(format="numpy"))
            for name in names
        ]

        for name, values in zip(names, before_ids, strict=True):
            assert len(np.unique(values)) == len(values), (
                f"{identifier!r} is not unique in dataset {name!r}"
            )
        for name, values in zip(names, after_ids, strict=True):
            assert len(np.unique(values)) == len(values), (
                f"{identifier!r} is not unique in dataset {name!r}"
            )

        before_pairs = set(zip(*before_ids, strict=True))
        after_pairs = set(zip(*after_ids, strict=True))
        assert before_pairs == after_pairs, f"Mapping differs for source {source!r}"


def _assert_ordered_mapping_equal(
    before, after, source, identifier="fof_halo_tag", sort_by=None
):
    """Assert exact matched row order survives persistence."""
    after = after.match(source)
    if sort_by is not None:
        after = after.sort_by(*sort_by).take_range(0, len(after[source]))
    assert tuple(before.keys()) == tuple(after.keys())
    for name in before.keys():
        np.testing.assert_array_equal(
            _column(after[name], identifier),
            _column(before[name], identifier),
            err_msg=name,
        )


@pytest.mark.parametrize("source", MAPPED_SIMULATIONS)
def test_match_aligns_rows_for_each_source(source, mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(source)

    assert isinstance(matched, oc.SimulationCollection)
    assert set(matched) == set(MAPPED_SIMULATIONS)
    _assert_matches_mapping(collection, matched, source, pairwise)


@pytest.mark.parametrize("source", (SIMULATION_A, SIMULATION_B))
def test_match_without_reference(source, mapped_paths, test_data):
    simulations = (SIMULATION_A, SIMULATION_B)
    collection = _open_mapped(
        mapped_paths, test_data.snapshot.halo_mapping, simulations
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(source)

    assert set(matched) == set(simulations)
    _assert_matches_mapping(collection, matched, source, pairwise)


def test_match_honors_existing_row_selection(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).take_range(
        10_000, 100_000
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(REFERENCE)

    _assert_matches_mapping(collection, matched, REFERENCE, pairwise)


@pytest.mark.parametrize(
    "filtered_simulations",
    (None, (REFERENCE,), (SIMULATION_A,)),
    ids=("all", "source-only", "target-only"),
)
def test_match_honors_filters(filtered_simulations, mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    original_indices = {
        name: into_array(dataset.index) for name, dataset in collection.items()
    }
    collection = collection.filter(
        oc.col("fof_halo_mass") > 1e14,
        datasets=filtered_simulations,
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    expected_filtered = set(filtered_simulations or MAPPED_SIMULATIONS)
    for name, dataset in collection.items():
        if name in expected_filtered:
            assert len(dataset) < len(original_indices[name])
        else:
            np.testing.assert_array_equal(
                into_array(dataset.index), original_indices[name]
            )

    matched = collection.match(REFERENCE)

    _assert_matches_mapping(collection, matched, REFERENCE, pairwise)


@pytest.mark.parametrize("source", MAPPED_SIMULATIONS)
def test_matched_take_range_is_driven_by_active_source(source, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(source)
    expected_source = matched[source].take_range(7, 31)

    result = matched.take_range(7, 31)

    _assert_source_driven_result(original, expected_source, result, source, pairwise)


def test_matched_filter_is_evaluated_only_on_active_source(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    threshold = np.median(_column(matched[REFERENCE], "fof_halo_mass"))
    mask = oc.col("fof_halo_mass") > threshold
    expected_source = matched[REFERENCE].filter(mask)

    result = matched.filter(mask)

    assert 0 < len(expected_source) < len(matched[REFERENCE])
    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parametrize("invert", (False, True), ids=("ascending", "descending"))
def test_matched_sort_uses_active_source_order(invert, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    expected_source = matched[REFERENCE].sort_by("fof_halo_mass", invert=invert)
    result = matched.sort_by("fof_halo_mass", invert=invert)

    masses = _column(result[REFERENCE], "fof_halo_mass")
    differences = np.diff(masses)
    assert np.all(differences <= 0 if invert else differences >= 0)
    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parametrize("at", ("start", "end"))
def test_matched_take_is_driven_by_active_source(at, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    expected_source = matched[REFERENCE].take(23, at=at)

    result = matched.take(23, at=at)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


def test_matched_random_take_preserves_source_order(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)

    result = matched.take(37, at="random")
    selected_source = result[REFERENCE]

    assert len(selected_source) == 37
    selected_ids = _column(selected_source, "fof_halo_tag")
    matched_ids = _column(matched[REFERENCE], "fof_halo_tag")
    positions = {value: position for position, value in enumerate(matched_ids.tolist())}
    selected_positions = np.asarray(
        [positions[value] for value in selected_ids.tolist()]
    )
    assert np.all(np.diff(selected_positions) > 0)
    _assert_source_driven_result(original, selected_source, result, REFERENCE, pairwise)


def test_matched_bound_is_evaluated_on_active_source(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE).with_units("scalefree")
    coordinate_names = tuple(f"fof_halo_center_{axis}" for axis in "xyz")
    coordinates = matched[REFERENCE].select(coordinate_names).get_data(format="numpy")
    lower = tuple(
        float(np.quantile(coordinates[name], 0.3)) for name in coordinate_names
    )
    upper = tuple(
        float(np.quantile(coordinates[name], 0.7)) for name in coordinate_names
    )
    region = oc.make_box(lower, upper)
    expected_source = matched[REFERENCE].bound(region)

    result = matched.bound(region)

    assert 0 < len(expected_source) < len(matched[REFERENCE])
    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


def test_matched_operation_intersects_pre_filtered_target(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    target_threshold = np.median(_column(original[SIMULATION_A], "fof_halo_mass"))
    original = original.filter(
        oc.col("fof_halo_mass") > target_threshold, datasets=SIMULATION_A
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    source_threshold = np.median(_column(matched[REFERENCE], "fof_halo_mass"))
    mask = oc.col("fof_halo_mass") > source_threshold
    expected_source = matched[REFERENCE].filter(mask)

    result = matched.filter(mask)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


def test_matched_chained_index_operations_remain_aligned(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(SIMULATION_A)
    threshold = np.median(_column(matched[SIMULATION_A], "fof_halo_mass"))
    mask = oc.col("fof_halo_mass") > threshold
    expected_source = (
        matched[SIMULATION_A]
        .filter(mask)
        .sort_by("fof_halo_mass", invert=True)
        .take_range(3, 29)
    )

    result = (
        matched.filter(mask).sort_by("fof_halo_mass", invert=True).take_range(3, 29)
    )

    _assert_source_driven_result(
        original, expected_source, result, SIMULATION_A, pairwise
    )


@pytest.mark.parametrize(
    "operation",
    ("filter", "sort", "take-start", "take-end", "take-random", "range", "bound"),
)
def test_clear_match_rebuilds_pending_targets(operation, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)

    if operation == "filter":
        result = matched.filter(oc.col("fof_halo_mass") > 1e14)
    elif operation == "sort":
        result = matched.sort_by("fof_halo_mass", invert=True)
    elif operation.startswith("take-"):
        result = matched.take(23, at=operation.removeprefix("take-"))
    elif operation == "range":
        result = matched.take_range(7, 31)
    else:
        matched = matched.with_units("scalefree")
        result = matched.bound(oc.make_box((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)))

    expected_source = result[REFERENCE]
    cleared = result.clear_match()

    _assert_source_driven_result(
        original, expected_source, cleared, REFERENCE, pairwise
    )
    assert {len(dataset) for dataset in cleared.values()} == {len(expected_source)}


def test_mapped_collection_context_manager_exits_cleanly(mapped_paths, test_data):
    with (
        _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
        .match(REFERENCE)
        .take_range(7, 31) as matched
    ):
        assert {len(dataset) for dataset in matched.values()} == {24}


def test_matched_targets_are_rebuilt_at_most_once(mapped_paths, test_data, monkeypatch):
    calls = 0
    prepare = simulation_module.prepare_matched_datasets

    def counting_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return prepare(*args, **kwargs)

    monkeypatch.setattr(simulation_module, "prepare_matched_datasets", counting_prepare)
    matched = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).match(
        REFERENCE
    )

    list(matched.values())
    list(matched.items())
    repr(matched)

    assert calls == 1

    pending = matched.take_range(7, 31)
    list(pending.values())
    list(pending.values())

    assert calls == 2


@pytest.mark.parametrize("accessor", ("getitem", "values", "items"))
def test_matched_dataset_access_rebuilds_targets(accessor, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE).take_range(7, 31)
    expected_source = matched[REFERENCE]
    expected_ids = _expected_source_driven_ids(
        original, expected_source, REFERENCE, pairwise
    )

    if accessor == "getitem":
        accessed = {name: matched[name] for name in matched.keys()}
    elif accessor == "values":
        accessed = dict(zip(matched.keys(), matched.values(), strict=True))
    else:
        accessed = dict(matched.items())

    for name, dataset in accessed.items():
        np.testing.assert_array_equal(
            _column(dataset, "fof_halo_tag"), expected_ids[name]
        )


def test_matched_evaluate_rebuilds_targets_before_evaluation(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE).take_range(7, 31)
    expected_source = matched[REFERENCE]
    expected_ids = _expected_source_driven_ids(
        original, expected_source, REFERENCE, pairwise
    )

    def evaluated_tag(fof_halo_tag):
        return fof_halo_tag

    result = matched.evaluate(
        evaluated_tag, vectorize=True, insert=False, format="numpy"
    )

    for name, dataset in matched.items():
        np.testing.assert_array_equal(
            _column(dataset, "fof_halo_tag"), expected_ids[name]
        )
        np.testing.assert_array_equal(
            result[name]["evaluated_tag"], _column(dataset, "fof_halo_tag")
        )


@pytest.mark.parametrize(
    "transform",
    (
        pytest.param(
            lambda collection: collection.select("fof_halo_tag", "fof_halo_mass"),
            id="column-selection",
        ),
        pytest.param(
            lambda collection: collection.with_units("scalefree"),
            id="unit-conversion",
        ),
        pytest.param(
            lambda collection: collection.with_new_columns(
                doubled_mass=oc.col("fof_halo_mass") * 2
            ),
            id="derived-column",
        ),
    ),
)
def test_non_index_operation_preserves_active_match_source(
    transform, mapped_paths, test_data
):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    expected_source = transform(matched[REFERENCE]).take_range(4, 19)

    result = transform(matched).take_range(4, 19)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parametrize("datasets", (REFERENCE, [REFERENCE], (REFERENCE,)))
def test_matched_filter_accepts_active_source_dataset_forms(
    datasets, mapped_paths, test_data
):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    collection = original.match(REFERENCE)
    mask = oc.col("fof_halo_mass") > 1e14
    expected_source = collection[REFERENCE].filter(mask)

    result = collection.filter(mask, datasets=datasets)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


def test_matched_filter_rejects_non_source_datasets(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).match(
        REFERENCE
    )
    mask = oc.col("fof_halo_mass") > 1e14

    with pytest.raises(ValueError, match="active source"):
        collection.filter(mask, datasets=SIMULATION_A)
    with pytest.raises(ValueError, match="active source"):
        collection.filter(mask, datasets=(REFERENCE, SIMULATION_A))


@pytest.mark.parametrize(
    ("paths", "message"),
    (
        (("primary", "mapping_reference"), "different simulations"),
        (("primary", "alternate_step"), "KAPPA_2_EGW_0.568_SEED_1.048e6"),
    ),
)
def test_open_without_connecting_mapping_raises(paths, message, test_data):
    snapshot = test_data.snapshot
    first = (
        snapshot.primary.halo_properties
        if paths[0] == "primary"
        else getattr(snapshot, paths[0])
    )
    second = getattr(snapshot, paths[1])

    with pytest.raises(ValueError, match=message):
        oc.open(first, second)


def test_mapping_file_alone_raises(test_data):
    with pytest.raises(ValueError, match="Cannot open a dataset mapping on its own"):
        oc.open(test_data.snapshot.halo_mapping)


def test_open_multiple_mapping_files_raises(mapped_paths, test_data, tmp_path):
    second_mapping = tmp_path / "second_mapping.hdf5"
    shutil.copy(test_data.snapshot.halo_mapping, second_mapping)

    with pytest.raises(ValueError, match="multiple dataset mapping files"):
        oc.open(
            mapped_paths[REFERENCE],
            mapped_paths[SIMULATION_A],
            test_data.snapshot.halo_mapping,
            second_mapping,
        )


def test_primary_mapping_length_must_match_reference(mapped_paths, test_data, tmp_path):
    mapping = tmp_path / "invalid_length_mapping.hdf5"
    shutil.copy(test_data.snapshot.halo_mapping, mapping)
    with h5py.File(mapping, "a") as file:
        primary = file["map/primary"]
        target = next(iter(primary))
        slot = primary[target]
        values = slot["index"][:-1]
        del slot["index"]
        slot.create_dataset("index", data=values)

    with pytest.raises(ValueError, match="reference dataset length"):
        oc.open(
            mapped_paths[REFERENCE],
            mapped_paths[SIMULATION_A],
            mapping,
        )


def test_match_requires_mapping(test_data):
    collection = oc.open(test_data.snapshot.multi_simulation)

    with pytest.raises(ValueError, match="does not contain matching information"):
        collection.match("scidac1")


def test_match_requires_known_source(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)

    with pytest.raises(ValueError, match="does not have a simulation named unknown"):
        collection.match("unknown")


def test_mapping_write(mapped_paths, test_data, tmp_path):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    collection = collection.filter(oc.col("fof_halo_mass") > 1e14)
    oc.write(tmp_path / "test.hdf5", collection)
    written = oc.open(tmp_path / "test.hdf5")

    _assert_mapping_equal(collection, written)


def test_mapping_write_without_reference(mapped_paths, test_data, tmp_path):
    simulations = (SIMULATION_A, SIMULATION_B)
    collection = _open_mapped(
        mapped_paths, test_data.snapshot.halo_mapping, simulations
    )
    collection = collection.filter(oc.col("fof_halo_mass") > 1e14)
    oc.write(tmp_path / "test.hdf5", collection)
    written = oc.open(tmp_path / "test.hdf5")

    _assert_mapping_equal(collection, written)


@pytest.mark.parametrize(
    ("transform", "sort_after_read"),
    (
        pytest.param(
            lambda collection: collection.filter(oc.col("fof_halo_mass") > 1e14),
            None,
            id="filter",
        ),
        pytest.param(
            lambda collection: collection.sort_by(
                "fof_halo_mass", invert=True
            ).take_range(7, 31),
            ("fof_halo_mass", True),
            id="sort-and-range",
        ),
        pytest.param(
            lambda collection: collection.take(37, at="random"),
            None,
            id="random-take",
        ),
    ),
)
def test_active_match_write_preserves_order(
    transform, sort_after_read, mapped_paths, test_data, tmp_path
):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).match(
        REFERENCE
    )
    collection = transform(collection)
    path = tmp_path / "test.hdf5"

    oc.write(path, collection)
    written = oc.open(path)

    _assert_ordered_mapping_equal(
        collection, written, REFERENCE, sort_by=sort_after_read
    )


def test_rebuild_with_new_source_preserves_old_reference_pair_as_auxiliary(tmp_path):
    old_reference, old_source, old_a, old_b = (uuid4() for _ in range(4))
    new_source, new_a, new_b = (uuid4() for _ in range(3))
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_source = file.create_dataset("source", data=[0, -1])
        primary_a = file.create_dataset("a", data=[-1, 0])
        primary_b = file.create_dataset("b", data=[-1, 0])
        match_set = DatasetMatchSet(
            old_reference,
            {old_source: primary_source, old_a: primary_a, old_b: primary_b},
            {},
            {"source": old_source, "a": old_a, "b": old_b},
        )

        primary, auxiliary = rebuild_single_with_new_source(
            match_set,
            {"source": new_source, "a": new_a, "b": new_b},
            {
                "source": np.array([0]),
                "a": np.array([0]),
                "b": np.array([0]),
            },
            "source",
        )

    np.testing.assert_array_equal(primary[new_a], [-1])
    np.testing.assert_array_equal(primary[new_b], [-1])
    np.testing.assert_array_equal(auxiliary[(new_a, new_b)][0], [0])
    np.testing.assert_array_equal(auxiliary[(new_a, new_b)][1], [0])
