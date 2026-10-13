"""
Characterization tests for the user-facing semantics of SimulationCollection.

These pin the mapping structure, child types, and dispatch to nested
collections that must survive the move to a SimulationState.
"""

import numpy as np
import pytest

import opencosmo as oc
from opencosmo.header import OpenCosmoHeader


@pytest.fixture
def multi_path(test_data):
    return test_data.snapshot.multi_simulation


@pytest.fixture
def structure_collection(test_data):
    children = {
        f"scidac_{i:03d}": oc.open(*test_data.snapshot.scidac(i).all) for i in (0, 1)
    }
    return _collection(children)


@pytest.fixture
def lightcone_children(test_data):
    return {
        f"step_{step}": oc.open(test_data.lightcone.step(step).halo_properties)
        for step in (600, 601)
    }


def _collection(children):
    return oc.SimulationCollection.from_datasets(children)


def test_dataset_children_mapping(multi_path):
    collection = oc.open(multi_path)
    keys = list(collection.keys())

    assert keys == ["scidac1", "scidac2"]
    assert len(collection) == 2
    assert list(collection) == keys
    assert [name for name, _ in collection.items()] == keys
    assert all(isinstance(value, oc.Dataset) for value in collection.values())
    assert all(isinstance(collection[name], oc.Dataset) for name in keys)
    assert collection.dtype == {name: "halo_properties" for name in keys}
    assert list(collection.header) == keys
    assert all(isinstance(h, OpenCosmoHeader) for h in collection.header.values())


def test_header_parameters_are_broadcast(multi_path):
    collection = oc.open(multi_path)
    header = collection.header["scidac1"]

    assert set(header.parameters) <= set(dir(collection))
    assert collection.simulation == {
        name: value.simulation for name, value in collection.header.items()
    }


def test_structure_children_mapping(structure_collection):
    keys = list(structure_collection.keys())

    assert keys == ["scidac_000", "scidac_001"]
    assert all(
        isinstance(value, oc.StructureCollection)
        for value in structure_collection.values()
    )
    assert structure_collection.dtype == {name: "halo_properties" for name in keys}
    for child in structure_collection.values():
        assert set(child.keys()) == {"galaxy_properties", "halo_properties"}


def test_structure_children_receive_mapped_operations(structure_collection):
    threshold = 1e13
    result = structure_collection.filter(oc.col("fof_halo_mass") > threshold).select(
        halo_properties=["fof_halo_mass", "fof_halo_tag"]
    )

    for child in result.values():
        assert isinstance(child, oc.StructureCollection)
        assert set(child["halo_properties"].columns) == {
            "fof_halo_mass",
            "fof_halo_tag",
        }
        masses = child["halo_properties"].get_data("numpy")["fof_halo_mass"]
        assert len(masses) > 0
        assert np.all(masses > threshold)


def test_structure_children_get_data_raises(structure_collection):
    with pytest.raises(TypeError, match="only supported when this collection"):
        structure_collection.get_data()


def test_structure_children_write_roundtrip(structure_collection, tmp_path):
    collection = structure_collection.take(50, at="start")
    path = tmp_path / "structures.hdf5"
    oc.write(path, collection)

    written = oc.open(path)
    assert isinstance(written, oc.SimulationCollection)
    assert set(written.keys()) == set(collection.keys())
    for name, child in written.items():
        assert isinstance(child, oc.StructureCollection)
        expected = collection[name]["halo_properties"].get_data("numpy")
        actual = child["halo_properties"].get_data("numpy")
        np.testing.assert_array_equal(
            np.sort(actual["fof_halo_tag"]), np.sort(expected["fof_halo_tag"])
        )


def test_unmatched_operation_restricted_to_named_dataset(multi_path):
    collection = oc.open(multi_path)
    threshold = 1e13

    result = collection.filter(oc.col("fof_halo_mass") > threshold, datasets="scidac1")

    assert len(result["scidac1"]) < len(collection["scidac1"])
    assert len(result["scidac2"]) == len(collection["scidac2"])


def test_unknown_dataset_raises(multi_path):
    collection = oc.open(multi_path)
    with pytest.raises(ValueError, match="Unknown datasets"):
        collection.filter(oc.col("fof_halo_mass") > 1e13, datasets="missing")


def test_operations_do_not_change_original(multi_path):
    collection = oc.open(multi_path)
    lengths = {name: len(value) for name, value in collection.items()}

    collection.take(10)
    collection.with_new_columns(doubled=oc.col("fof_halo_mass") * 2)

    assert {name: len(value) for name, value in collection.items()} == lengths
    assert all("doubled" not in value.columns for value in collection.values())


def test_lightcone_children(lightcone_children):
    collection = _collection(lightcone_children)

    assert list(collection.keys()) == ["step_600", "step_601"]
    assert all(isinstance(value, oc.Lightcone) for value in collection.values())

    threshold = 1e13
    result = collection.filter(oc.col("fof_halo_mass") > threshold)
    data = result.get_data("numpy")
    for name, child in lightcone_children.items():
        assert len(data[name]["fof_halo_mass"]) < len(child)
        assert np.all(data[name]["fof_halo_mass"] > threshold)


def test_context_manager_with_structure_children(structure_collection):
    with structure_collection as collection:
        assert len(collection) == 2


def test_lightcone_children_column_operations(lightcone_children):
    collection = _collection(lightcone_children)

    selected = collection.select("fof_halo_mass", "fof_halo_tag")
    dropped = selected.drop("fof_halo_tag")

    for name in collection:
        assert isinstance(selected[name], oc.Lightcone)
        assert set(selected[name].columns) == {"fof_halo_mass", "fof_halo_tag"}
        assert dropped[name].columns == ["fof_halo_mass"]
        assert len(dropped[name]) == len(lightcone_children[name])


def test_take_rows_applies_to_every_member(multi_path):
    collection = oc.open(multi_path)
    rows = np.array([0, 3, 8, 21], dtype=np.int64)

    result = collection.take_rows(rows)

    for name, child in result.items():
        expected = collection[name].take_rows(rows)
        np.testing.assert_array_equal(
            child.select("fof_halo_tag").get_data("numpy"),
            expected.select("fof_halo_tag").get_data("numpy"),
        )


def test_take_rows_out_of_range_raises(multi_path):
    collection = oc.open(multi_path)
    too_far = max(len(child) for child in collection.values())
    with pytest.raises(ValueError, match="Row indices must be between"):
        collection.take_rows(np.array([0, too_far], dtype=np.int64))


def test_from_datasets_rejects_unsupported_members(multi_path):
    dataset = next(iter(oc.open(multi_path).values()))
    with pytest.raises(TypeError, match="only accepts datasets"):
        oc.SimulationCollection.from_datasets({"a": dataset, "b": object()})
