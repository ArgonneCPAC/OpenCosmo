from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from uuid import uuid4

import h5py
import numpy as np
import pytest

import opencosmo as oc
from opencosmo.index import into_array
from opencosmo.mapping.mapping import DatasetMatchSet, rebuild_single_with_new_source

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
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping, simulations)
    collection = collection.filter(oc.col("fof_halo_mass") > 1e14)
    oc.write(tmp_path / "test.hdf5", collection)
    written = oc.open(tmp_path / "test.hdf5")

    _assert_mapping_equal(collection, written)


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
