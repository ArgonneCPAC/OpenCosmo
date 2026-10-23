"""
Pure tests for StructureState and its deferred rebuilding rule.
"""

from dataclasses import replace

import numpy as np
import opencosmo as oc
import pytest
from opencosmo.collection.structure import links
from opencosmo.collection.structure import state as scst
from opencosmo.dataset import operations as dsops
from opencosmo.io.io import open_state


def _tags(state):
    return oc.Dataset(state).select("fof_halo_tag").get_data("numpy")


def _run_values(values):
    return values[np.r_[0, np.flatnonzero(np.diff(values)) + 1]]


@pytest.fixture
def halo_paths(test_data):
    return test_data.snapshot.primary.halos


@pytest.fixture
def galaxy_paths(test_data):
    return test_data.snapshot.primary.galaxies


@pytest.fixture
def halo_state(halo_paths):
    return open_state(*halo_paths)


@pytest.fixture
def derived_state(halo_state):
    new_source = dsops.take_rows(halo_state.source, np.arange(5))
    return replace(halo_state, source=new_source, linked_source=halo_state.source)


def test_children_without_linked_source_is_linked(halo_state):
    assert halo_state.linked_source is None
    assert halo_state.children is halo_state.linked


def test_children_computed_once(derived_state):
    children = derived_state.children
    assert children is not derived_state.linked
    assert derived_state.children is children


def test_replace_does_not_carry_children(derived_state):
    _ = derived_state.children
    assert "children" in derived_state.__dict__
    copied = replace(derived_state)
    assert "children" not in copied.__dict__
    assert copied.children is not derived_state.children


def test_children_rebuild_against_new_source(derived_state):
    particle_tags = _tags(derived_state.children["dm_particles"])
    assert np.array_equal(_run_values(particle_tags), _tags(derived_state.source))


def test_take_per_leaf_structure_defers_rebuild(halo_state):
    taken = links.take_per_leaf(halo_state, {(): np.arange(5)})
    assert isinstance(taken, scst.StructureState)
    assert taken.linked_source is halo_state.source
    assert taken.linked is halo_state.linked
    assert len(taken) == 5
    particle_tags = _tags(taken.children["dm_particles"])
    assert np.array_equal(_run_values(particle_tags), _tags(taken.source))


def test_take_per_leaf_structure_keeps_original_linked_source(derived_state):
    taken = links.take_per_leaf(derived_state, {(): np.arange(2)})
    assert taken.linked_source is derived_state.linked_source
    assert len(taken) == 2


def test_read_only_properties(halo_paths, halo_state):
    collection = oc.open(*halo_paths)
    assert len(halo_state) == len(collection)
    assert halo_state.dtype == "halo_properties"
    assert halo_state.header is halo_state.source.header
    assert halo_state.properties == collection.properties
    assert halo_state.sorted_by is None
    sorted_state = replace(
        halo_state, source=dsops.sort_by(halo_state.source, "fof_halo_mass", False)
    )
    assert sorted_state.sorted_by == "fof_halo_mass"


def test_public_keys_match_facade(halo_paths, galaxy_paths):
    for paths in (halo_paths, [*halo_paths, *galaxy_paths]):
        assert scst.public_keys(open_state(*paths)) == oc.open(*paths).keys()


def test_public_keys_hide_source(halo_state):
    hidden = replace(halo_state, hide_source=True)
    assert "halo_properties" not in scst.public_keys(hidden)
    assert scst.view(hidden, "halo_properties") is hidden.source


def test_view(derived_state):
    assert scst.view(derived_state, "halo_properties") is derived_state.source
    assert (
        scst.view(derived_state, "dm_particles")
        is (derived_state.children["dm_particles"])
    )
    with pytest.raises(KeyError):
        scst.view(derived_state, "not_a_dataset")


def test_from_parts_renames_nested_galaxies(halo_paths, galaxy_paths):
    state = open_state(*halo_paths, *galaxy_paths)
    linked = dict(state.linked)
    linked["galaxy_properties"] = linked.pop("galaxies")
    rebuilt = scst.from_parts(state.source, linked, state.match_sets, resolve=False)
    assert "galaxy_properties" not in rebuilt.linked
    assert isinstance(rebuilt.linked["galaxies"], scst.StructureState)


def test_from_parts_keeps_galaxy_properties_dataset(halo_paths, galaxy_paths):
    state = open_state(halo_paths[0], galaxy_paths[0])
    assert list(state.linked) == ["galaxy_properties"]


def test_from_parts_requires_match_sets(halo_state):
    with pytest.raises(ValueError, match="match sets"):
        scst.from_parts(halo_state.source, halo_state.linked, {}, resolve=False)


def test_from_parts_without_links_needs_no_match_sets(halo_state):
    state = scst.from_parts(halo_state.source, {}, {}, resolve=False)
    assert scst.public_keys(state) == ["halo_properties"]
