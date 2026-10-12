import pytest
from opencosmo.dataset.state import DatasetState

import opencosmo as oc
from opencosmo.collection.lightcone import state as lcst


@pytest.fixture
def plain_leaves(test_data):
    lc = oc.open(
        test_data.lightcone.step(601).halo_properties,
        test_data.lightcone.step(600).halo_properties,
    )
    return tuple(((step,), ds._state) for step, ds in lc.items())


@pytest.fixture
def nested_leaves(test_data):
    lc = oc.open(
        test_data.diffsky.core(475), test_data.diffsky.core(487), synth_cores=True
    )
    return tuple(
        ((step, subtype), ds._state)
        for step, inner in lc.items()
        for subtype, ds in inner.items()
    )


def test_from_leaves_rejects_empty():
    with pytest.raises(ValueError, match="at least one dataset"):
        lcst.from_leaves(())


def test_from_leaves_rejects_mixed_key_depth(plain_leaves):
    (key, leaf), *rest = plain_leaves
    with pytest.raises(ValueError, match="same depth"):
        lcst.from_leaves(((key + ("data",), leaf), *rest))


def test_from_leaves_rejects_mismatched_columns(plain_leaves):
    from opencosmo.dataset.state import select

    (key, leaf), *rest = plain_leaves
    narrowed = select(leaf, {"fof_halo_mass"})
    with pytest.raises(ValueError, match="same columns"):
        lcst.from_leaves(((key, narrowed), *rest))


def test_from_leaves_computes_z_range(nested_leaves):
    state = lcst.from_leaves(nested_leaves)
    leaf_ranges = [lcst.header_z_range(leaf.header) for _, leaf in nested_leaves]
    assert state.z_range == (
        min(r[0] for r in leaf_ranges),
        max(r[1] for r in leaf_ranges),
    )
    assert lcst.from_leaves(nested_leaves, z_range=(0.0, 0.01)).z_range == (0.0, 0.01)


def test_state_length_and_columns(plain_leaves):
    state = lcst.from_leaves(plain_leaves, hidden=frozenset({"ra", "dec"}))
    assert len(state) == sum(len(leaf) for _, leaf in plain_leaves)
    assert "ra" not in state.columns and "dec" not in state.columns
    assert "ra" not in state.units and "ra" not in state.descriptions
    assert state.uuid == plain_leaves[0][1].uuid


def test_public_keys_plain(plain_leaves):
    state = lcst.from_leaves(plain_leaves)
    assert lcst.public_keys(state) == [600, 601]
    assert all(isinstance(lcst.view(state, k), DatasetState) for k in (600, 601))


def test_public_keys_nested(nested_leaves):
    state = lcst.from_leaves(nested_leaves)
    assert lcst.public_keys(state) == [487, 475]

    step = lcst.view(state, 487)
    assert isinstance(step, lcst.LightconeState)
    assert lcst.public_keys(step) == ["cores", "synth_cores"]
    assert isinstance(lcst.view(step, "cores"), DatasetState)
    assert step.z_range[1] < state.z_range[1]


def test_view_with_single_remaining_subtype_is_lightcone(nested_leaves):
    cores_only = tuple((k, leaf) for k, leaf in nested_leaves if k[1] == "cores")
    state = lcst.from_leaves(cores_only)
    step = lcst.view(state, 487)
    assert isinstance(step, lcst.LightconeState)
    assert lcst.public_keys(step) == ["cores"]


def test_view_drops_lightcone_level_settings(nested_leaves):
    state = lcst.from_leaves(
        nested_leaves, hidden=frozenset({"ra"}), sort_key=("logsm_obs", False)
    )
    step = lcst.view(state, 487)
    assert isinstance(step, lcst.LightconeState)
    assert step.hidden == frozenset()
    assert step.sort_key is None
    assert step.maps is None
    assert "ra" in step.columns


def test_view_missing_key(plain_leaves):
    with pytest.raises(KeyError):
        lcst.view(lcst.from_leaves(plain_leaves), 9999)
