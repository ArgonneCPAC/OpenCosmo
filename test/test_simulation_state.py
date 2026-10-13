"""
Pure tests for SimulationState and its deferred realignment rule.
"""

from dataclasses import replace

import numpy as np
import pytest

import opencosmo as oc
from opencosmo.collection.simulation import matching
from opencosmo.collection.simulation import state as simst
from opencosmo.dataset import operations as dsops
from opencosmo.utils import normalize_kwarg_name

REFERENCE = normalize_kwarg_name("SCIDAC_128_GO")


@pytest.fixture
def mapped(test_data):
    snapshot = test_data.snapshot
    return oc.open(
        snapshot.mapping_reference,
        *(snapshot.scidac(i).halo_properties for i in range(3)),
        snapshot.halo_mapping,
    )


@pytest.fixture
def match_set(mapped):
    return mapped._state.match_set


@pytest.fixture
def matched_state(mapped, match_set):
    members = {
        name: dataset._state for name, dataset in mapped.match(REFERENCE).items()
    }
    return simst.SimulationState(members, match_set, REFERENCE)


@pytest.fixture
def pending_state(matched_state):
    source = matched_state.members[REFERENCE]
    new_source = dsops.take_range(source, 7, 31, "local")
    return replace(
        matched_state,
        members=matched_state.members | {REFERENCE: new_source},
        aligned=False,
    )


def _tags(state):
    return oc.Dataset(state).select("fof_halo_tag").get_data("numpy")


def test_aligned_children_are_members(matched_state):
    assert matched_state.aligned
    assert matched_state.children is matched_state.members


def test_children_computed_once(pending_state):
    children = pending_state.children
    assert children is not pending_state.members
    assert pending_state.children is children


def test_replace_does_not_carry_children(pending_state):
    children = pending_state.children
    copy = replace(pending_state)
    assert "children" not in copy.__dict__
    assert copy.children is not children


def test_pending_children_match_facade(mapped, pending_state):
    expected = mapped.match(REFERENCE).take_range(7, 31)

    children = pending_state.children

    assert list(children) == list(pending_state.members)
    assert {len(child) for child in children.values()} == {24}
    for name, child in children.items():
        np.testing.assert_array_equal(_tags(child), _tags(expected[name]._state))


def test_align_preserves_member_order(match_set, matched_state):
    members = dict(reversed(list(matched_state.members.items())))
    aligned = matching.align(match_set, REFERENCE, members)
    assert list(aligned) == list(members)


def test_align_rejects_collection_members(match_set, matched_state, test_data):
    structure = oc.open(*test_data.snapshot.scidac(0).all)._state
    members = matched_state.members | {"structure": structure}
    with pytest.raises(RuntimeError, match="is not a dataset"):
        matching.align(match_set, REFERENCE, members)


def test_from_parts_normalizes_names(mapped):
    dataset = next(iter(mapped.values()))._state
    state = simst.from_parts({"sim.one": dataset})
    assert simst.public_keys(state) == ["sim_one"]
    assert state.match_source is None
    assert state.aligned


def test_from_parts_rejects_matching_collections(match_set, test_data):
    structure = oc.open(*test_data.snapshot.scidac(0).all)._state
    with pytest.raises(ValueError, match="only supported for simple datasets"):
        simst.from_parts({"structure": structure}, match_set)


@pytest.mark.parametrize(
    ("with_match_set", "kwargs", "message"),
    (
        (False, {"match_source": REFERENCE}, "requires a match set"),
        (True, {"match_source": "missing"}, "is not a member"),
        (False, {"aligned": False}, "Only a matched collection"),
    ),
)
def test_invalid_match_fields_raise(matched_state, with_match_set, kwargs, message):
    match_set = matched_state.match_set if with_match_set else None
    with pytest.raises(RuntimeError, match=message):
        simst.SimulationState(matched_state.members, match_set, **kwargs)


def test_view(pending_state):
    view = simst.view(pending_state, REFERENCE)
    assert view is pending_state.children[REFERENCE]
    with pytest.raises(KeyError, match="not found"):
        simst.view(pending_state, "missing")


def test_header_and_dtype_read_members(pending_state):
    assert set(pending_state.header) == set(pending_state.members)
    assert set(pending_state.dtype.values()) == {"halo_properties"}
    # Reading metadata does not realign the members.
    assert "children" not in pending_state.__dict__
