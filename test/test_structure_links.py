"""
Focused tests for state-based structure link arithmetic.
"""

import numpy as np
import opencosmo as oc
import pytest
from opencosmo.collection.lightcone import state as lcst
from opencosmo.collection.structure import links
from opencosmo.collection.structure import state as scst
from opencosmo.facade import wrap
from opencosmo.io.io import open_state


@pytest.fixture
def halo_paths(test_data):
    return test_data.snapshot.primary.halos


@pytest.fixture
def galaxy_paths(test_data):
    return test_data.snapshot.primary.galaxies


@pytest.fixture
def plain_lightcone(test_data):
    return open_state(
        test_data.lightcone.step(600).halo_properties,
        test_data.lightcone.step(601).halo_properties,
    )


def _tags(state):
    return wrap(state).select("fof_halo_tag").get_data("numpy")


def test_take_per_leaf_dataset(halo_paths):
    dataset = open_state(halo_paths[0])
    rows = np.array([0, 2, 4])
    taken = links.take_per_leaf(dataset, {(): rows})
    assert len(taken) == 3
    assert np.array_equal(_tags(taken), _tags(dataset)[rows])


def test_take_per_leaf_lightcone(plain_lightcone):
    taken = links.take_per_leaf(
        plain_lightcone, {(600,): np.arange(3), (601,): np.arange(2)}
    )
    assert [key for key, _ in taken.leaves] == [(600,), (601,)]
    assert [len(leaf) for _, leaf in taken.leaves] == [3, 2]
    expected = np.concatenate(
        [
            _tags(lcst.view(plain_lightcone, 600))[:3],
            _tags(lcst.view(plain_lightcone, 601))[:2],
        ]
    )
    assert np.array_equal(_tags(taken), expected)


def test_take_per_leaf_lightcone_drops_absent_steps(plain_lightcone):
    taken = links.take_per_leaf(plain_lightcone, {(601,): np.arange(2)})
    assert [key for key, _ in taken.leaves] == [(601,)]
    assert len(taken) == 2


def test_take_per_leaf_lightcone_keeps_empty_steps(plain_lightcone):
    empty = np.array([], dtype=np.int64)
    taken = links.take_per_leaf(plain_lightcone, {(600,): empty, (601,): empty})
    assert [key for key, _ in taken.leaves] == [(600,), (601,)]
    assert len(taken) == 0


def test_take_per_leaf_lightcone_missing_step_fails(plain_lightcone):
    with pytest.raises(RuntimeError):
        links.take_per_leaf(plain_lightcone, {(599,): np.arange(2)})


def test_take_per_leaf_nested_collection(halo_paths, galaxy_paths):
    galaxies = scst.view(open_state(*halo_paths, *galaxy_paths), "galaxies")
    assert isinstance(galaxies, scst.StructureState)
    taken = links.take_per_leaf(galaxies, {(): np.arange(3)})
    assert isinstance(taken, scst.StructureState)
    assert len(taken) == 3
    assert len(oc.StructureCollection(taken)) == 3


def test_stacked_index_offsets_lightcone_leaves(plain_lightcone):
    n_600 = len(lcst.view(plain_lightcone, 600))
    index = links.stacked_index(
        plain_lightcone, {(600,): np.array([0, 1]), (601,): np.array([0])}
    )
    assert np.array_equal(index, [0, 1, n_600])


def test_stacked_index_skips_absent_leaves(plain_lightcone):
    n_600 = len(lcst.view(plain_lightcone, 600))
    index = links.stacked_index(plain_lightcone, {(601,): np.array([1, 2])})
    assert np.array_equal(index, [n_600 + 1, n_600 + 2])
