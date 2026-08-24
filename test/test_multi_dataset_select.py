from __future__ import annotations

import numpy as np
import pytest
from opencosmo.column.select import (
    MissingColumnError,
    build_multi_dataset_selections,
)

import opencosmo as oc


@pytest.fixture
def dataset_columns():
    return {
        "gravity": {"id", "mass", "x"},
        "hydro": {"id", "mass", "temperature"},
    }


def test_routes_columns_and_wildcards(dataset_columns):
    args, _ = build_multi_dataset_selections(
        dataset_columns,
        {"gravity": 3, "hydro": 5},
        ("x", "temperature", "m*"),
        {},
    )

    assert set(args["gravity"]) == {"x", "m*"}
    assert set(args["hydro"]) == {"temperature", "m*"}


def test_routes_derived_columns_by_dependencies(dataset_columns):
    _, kwargs = build_multi_dataset_selections(
        dataset_columns,
        {"gravity": 3, "hydro": 5},
        (),
        {
            "mass_squared": oc.col("mass") ** 2,
            "thermal_mass": oc.col("mass") * oc.col("temperature"),
        },
    )

    assert set(kwargs["gravity"]) == {"mass_squared"}
    assert set(kwargs["hydro"]) == {"mass_squared", "thermal_mass"}


def test_routes_arrays_by_dataset_length(dataset_columns):
    gravity_values = np.arange(3)
    hydro_values = np.arange(5)
    _, kwargs = build_multi_dataset_selections(
        dataset_columns,
        {"gravity": 3, "hydro": 5},
        (),
        {"gravity_rank": gravity_values, "hydro_rank": hydro_values},
    )

    assert kwargs["gravity"]["gravity_rank"] is gravity_values
    assert kwargs["hydro"]["hydro_rank"] is hydro_values


def test_rejects_array_that_matches_no_dataset_length(dataset_columns):
    with pytest.raises(ValueError, match="rank"):
        build_multi_dataset_selections(
            dataset_columns,
            {"gravity": 3, "hydro": 5},
            (),
            {"rank": np.arange(4)},
        )


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("missing",), {}),
        ((), {"derived": oc.col("missing") * 2}),
    ],
)
def test_rejects_selections_missing_from_every_dataset(dataset_columns, args, kwargs):
    with pytest.raises(MissingColumnError):
        build_multi_dataset_selections(
            dataset_columns,
            {"gravity": 3, "hydro": 5},
            args,
            kwargs,
        )


def test_simulation_collection_routes_across_different_column_sets(test_data):
    gravity = oc.open(test_data.snapshot.primary.halo_properties)
    hydro = oc.open(test_data.snapshot.primary.galaxy_properties)
    collection = oc.SimulationCollection({"gravity": gravity, "hydro": hydro})

    selected = collection.select(
        "fof_halo_mass",
        "gal_mass_star",
        stellar_fraction=oc.col("gal_mass_star") / oc.col("gal_mass"),
    )

    assert set(selected["gravity"].columns) == {"fof_halo_mass"}
    assert set(selected["hydro"].columns) == {"gal_mass_star", "stellar_fraction"}
    hydro_data = selected["hydro"].get_data("numpy")
    expected = hydro.select(
        value=oc.col("gal_mass_star") / oc.col("gal_mass")
    ).get_data("numpy")
    assert np.all(hydro_data["stellar_fraction"] == expected)


def test_simulation_collection_drop_routes_across_different_column_sets(test_data):
    gravity = oc.open(test_data.snapshot.primary.halo_properties)
    hydro = oc.open(test_data.snapshot.primary.galaxy_properties)
    collection = oc.SimulationCollection({"gravity": gravity, "hydro": hydro})

    dropped = collection.drop("fof_halo_mass", "gal_mass_star")

    assert "fof_halo_mass" not in dropped["gravity"].columns
    assert "gal_mass_star" not in dropped["hydro"].columns
    assert "gal_mass" in dropped["hydro"].columns


def test_simulation_collection_drop_wildcard_leaves_unmatched_datasets_unchanged(
    test_data,
):
    gravity = oc.open(test_data.snapshot.primary.halo_properties)
    hydro = oc.open(test_data.snapshot.primary.galaxy_properties)
    collection = oc.SimulationCollection({"gravity": gravity, "hydro": hydro})

    dropped = collection.drop("gal_mass_*")

    assert dropped["gravity"].columns == gravity.columns
    assert not any(name.startswith("gal_mass_") for name in dropped["hydro"].columns)


def test_simulation_collection_drop_rejects_missing_column(test_data):
    gravity = oc.open(test_data.snapshot.primary.halo_properties)
    hydro = oc.open(test_data.snapshot.primary.galaxy_properties)
    collection = oc.SimulationCollection({"gravity": gravity, "hydro": hydro})

    with pytest.raises(MissingColumnError):
        collection.drop("not_a_column")


def test_simulation_collection_drop_by_dataset_key(test_data):
    gravity = oc.open(test_data.snapshot.primary.halo_properties)
    hydro = oc.open(test_data.snapshot.primary.galaxy_properties)
    collection = oc.SimulationCollection({"gravity": gravity, "hydro": hydro})

    dropped = collection.drop(gravity=["fof_halo_mass"], hydro=["gal_mass_star"])

    assert "fof_halo_mass" not in dropped["gravity"].columns
    assert "gal_mass_star" not in dropped["hydro"].columns


def test_structure_collection_ignores_unmatched_wildcard_on_other_datasets(test_data):
    collection = oc.open(*test_data.snapshot.primary.halos)
    selected = collection.select("sod_halo_mass*")

    assert selected["halo_properties"].columns
    assert all(
        name.startswith("sod_halo_mass") for name in selected["halo_properties"].columns
    )
    assert selected["dm_particles"].columns == collection["dm_particles"].columns


def test_structure_collection_rejects_scalar_selection(test_data):
    collection = oc.open(*test_data.snapshot.primary.halos)

    with pytest.raises(ValueError, match="Scalar values cannot be retrieved"):
        collection.select(mean_mass=oc.col("fof_halo_mass").mean())
