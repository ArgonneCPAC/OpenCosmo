"""
Characterization tests for the user-facing structure of Lightcone objects.

These pin the mapping semantics (keys, values, nesting, and empty-child
handling) that must survive the move to a flat LightconeState.
"""

import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def plain_lightcone(test_data):
    return oc.open(
        test_data.lightcone.step(601).halo_properties,
        test_data.lightcone.step(600).halo_properties,
    )


@pytest.fixture
def nested_lightcone(test_data):
    return oc.open(
        test_data.diffsky.core(475), test_data.diffsky.core(487), synth_cores=True
    )


def test_plain_lightcone_children_are_datasets(plain_lightcone):
    assert list(plain_lightcone.keys()) == [600, 601]
    assert all(isinstance(child, oc.Dataset) for child in plain_lightcone.values())
    assert len(plain_lightcone) == sum(len(c) for c in plain_lightcone.values())


def test_nested_lightcone_children_are_lightcones(nested_lightcone):
    assert list(nested_lightcone.keys()) == [487, 475]
    for step in nested_lightcone.values():
        assert isinstance(step, oc.Lightcone)
        assert list(step.keys()) == ["cores", "synth_cores"]
        assert all(isinstance(leaf, oc.Dataset) for leaf in step.values())


def test_nested_step_z_range_covers_only_its_step(nested_lightcone):
    step_ranges = [step.z_range for step in nested_lightcone.values()]
    assert step_ranges[0][1] == step_ranges[1][0]
    assert nested_lightcone.z_range == (step_ranges[0][0], step_ranges[1][1])


def test_stacking_order_follows_steps_then_subtypes(nested_lightcone):
    stacked = nested_lightcone.select("logsm_obs").get_data("numpy")
    pieces = [
        leaf.select("logsm_obs").get_data("numpy")
        for step in nested_lightcone.values()
        for leaf in step.values()
    ]
    assert np.array_equal(stacked, np.concatenate(pieces))


def test_nested_step_with_one_remaining_subtype_is_still_lightcone(
    nested_lightcone,
):
    cores_only = nested_lightcone.filter(oc.col("core_tag") >= 0)
    assert list(cores_only.keys()) == [487, 475]
    for step in cores_only.values():
        assert isinstance(step, oc.Lightcone)
        assert list(step.keys()) == ["cores"]


def test_take_rows_drops_empty_subtypes(nested_lightcone):
    first_step = nested_lightcone[487]
    n_cores = len(first_step["cores"])
    taken = nested_lightcone.take_rows(np.arange(n_cores))
    assert list(taken.keys()) == [487]
    assert isinstance(taken[487], oc.Lightcone)
    assert list(taken[487].keys()) == ["cores"]


def test_with_redshift_range_leaving_one_nested_step(nested_lightcone):
    restricted = nested_lightcone.with_redshift_range(0.0, 0.02)
    assert list(restricted.keys()) == [487]
    assert isinstance(restricted[487], oc.Lightcone)
    assert list(restricted[487].keys()) == ["cores", "synth_cores"]
    assert restricted.z_range == (0.0, 0.02)


def test_filter_drops_empty_steps(plain_lightcone):
    filtered = plain_lightcone.filter(oc.col("redshift") < 0.0395)
    assert list(filtered.keys()) == [601]
    assert len(filtered) > 0


def test_filter_to_empty_keeps_all_steps(plain_lightcone, nested_lightcone):
    empty = plain_lightcone.filter(oc.col("redshift") > 100)
    assert list(empty.keys()) == [600, 601]
    assert len(empty) == 0

    empty_nested = nested_lightcone.filter(oc.col("redshift_true") > 100)
    assert list(empty_nested.keys()) == [487, 475]
    for step in empty_nested.values():
        assert list(step.keys()) == ["cores", "synth_cores"]
    assert len(empty_nested) == 0


def test_take_rows_to_empty_keeps_first_step(plain_lightcone):
    empty = plain_lightcone.take_rows(np.array([], dtype=np.int64))
    assert list(empty.keys()) == [600]
    assert len(empty) == 0


def test_nested_step_does_not_inherit_scope_sort_or_hidden(nested_lightcone):
    mass = oc.col("logsm_obs")
    scoped = nested_lightcone.with_new_columns(centered=mass - mass.mean())
    assert "centered" in scoped.columns
    assert "centered" not in scoped[487].columns

    sorted_lc = nested_lightcone.sort_by("logsm_obs")
    assert sorted_lc.sorted_by == "logsm_obs"
    assert sorted_lc[487].sorted_by is None

    selected = nested_lightcone.select("logsm_obs")
    assert selected.columns == ["logsm_obs"]
    assert set(selected[487].columns) == {"logsm_obs", "ra", "dec", "redshift"}


def test_plain_child_exposes_hidden_required_columns(plain_lightcone):
    selected = plain_lightcone.select("fof_halo_mass")
    assert selected.columns == ["fof_halo_mass"]
    child = next(iter(selected.values()))
    assert set(child.columns) == {"fof_halo_mass", "ra", "dec", "redshift"}


def test_nested_evaluate_noinsert_broadcasts_mapped_kwargs(nested_lightcone):
    weights = {step: i + 1 for i, step in enumerate(nested_lightcone.keys())}

    def weight(logsm_obs, w):
        return {"y": np.full(len(logsm_obs), w)}

    result = nested_lightcone.evaluate(
        weight, vectorize=True, insert=False, format="numpy", w=weights
    )
    step_lengths = [len(step) for step in nested_lightcone.values()]
    assert np.array_equal(result["y"], np.repeat(list(weights.values()), step_lengths))


def test_header_follows_first_remaining_step(plain_lightcone):
    filtered = plain_lightcone.filter(oc.col("redshift") < 0.0395)
    assert filtered.header.file.step == 601
    assert filtered.z_range == plain_lightcone.z_range
