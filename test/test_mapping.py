from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

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
        auxiliary_a = auxiliary["source/index"][:]
        auxiliary_b = auxiliary["target/index"][:]

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


def test_match_requires_mapping(test_data):
    collection = oc.open(test_data.snapshot.multi_simulation)

    with pytest.raises(ValueError, match="does not contain matching information"):
        collection.match("scidac1")


def test_match_requires_known_source(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)

    with pytest.raises(ValueError, match="does not have a simulation named unknown"):
        collection.match("unknown")
