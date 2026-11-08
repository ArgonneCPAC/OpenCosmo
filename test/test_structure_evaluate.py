"""
Tests for StructureCollection.evaluate with column names as function arguments.
"""

import numpy as np
import opencosmo as oc
import pytest
from opencosmo.collection.structure.evaluate import resolve_column_arguments
from opencosmo.column.select import MissingColumnError

COLUMNS = {
    ("halo_properties",): {"fof_halo_mass", "fof_halo_tag"},
    ("dm_particles",): {"x", "y", "z", "fof_halo_tag"},
    ("gas_particles",): {"x", "y", "z", "rho"},
    ("galaxies", "galaxy_properties"): {"gal_mass_star", "fof_halo_tag"},
}


def test_resolve_columns_to_datasets():
    def f(fof_halo_mass, rho, gal_mass_star):
        pass

    assert resolve_column_arguments(f, COLUMNS, {}) == {
        "fof_halo_mass": ("halo_properties",),
        "rho": ("gas_particles",),
        "gal_mass_star": ("galaxies", "galaxy_properties"),
    }


def test_resolve_ambiguous_column_raises():
    def f(fof_halo_mass, x):
        pass

    with pytest.raises(ValueError, match="x was found in multiple datasets"):
        resolve_column_arguments(f, COLUMNS, {})


def test_resolve_ambiguous_nested_column_raises():
    def f(fof_halo_tag):
        pass

    with pytest.raises(ValueError, match="galaxies.galaxy_properties"):
        resolve_column_arguments(f, COLUMNS, {})


def test_resolve_kwargs_are_not_columns():
    def f(fof_halo_mass, scale):
        pass

    resolved = resolve_column_arguments(f, COLUMNS, {"scale": 2.0})
    assert resolved == {"fof_halo_mass": ("halo_properties",)}


def test_resolve_kwarg_colliding_with_column_raises():
    def f(fof_halo_mass, rho):
        pass

    with pytest.raises(ValueError, match="rho was passed to evaluate"):
        resolve_column_arguments(f, COLUMNS, {"rho": 2.0})


def test_resolve_missing_required_column_raises():
    def f(fof_halo_mass, not_a_column):
        pass

    with pytest.raises(MissingColumnError, match="not_a_column"):
        resolve_column_arguments(f, COLUMNS, {})


def test_resolve_optional_parameter_keeps_default():
    def f(fof_halo_mass, factor=2, *args, **kwargs):
        pass

    assert resolve_column_arguments(f, COLUMNS, {}) == {
        "fof_halo_mass": ("halo_properties",)
    }


@pytest.fixture
def halo_paths(test_data):
    return test_data.snapshot.primary.halos


@pytest.fixture
def galaxy_paths(test_data):
    return test_data.snapshot.primary.galaxies


@pytest.fixture
def halos(halo_paths):
    return oc.open(*halo_paths).take(50)


def test_evaluate_infers_datasets(halos):
    def gas_per_mass(fof_halo_mass, rho):
        return len(rho) / fof_halo_mass

    def explicit(halo_properties, gas_particles):
        return len(gas_particles["rho"]) / halo_properties["fof_halo_mass"]

    inferred = halos.evaluate(gas_per_mass, insert=False)
    expected = halos.evaluate(
        explicit,
        insert=False,
        halo_properties=["fof_halo_mass"],
        gas_particles=["rho"],
    )
    assert np.all(inferred["gas_per_mass"] == expected["explicit"])


def test_evaluate_inferred_insert(halos):
    def gas_count(fof_halo_tag, rho, scale):
        return scale * len(rho)

    with pytest.raises(ValueError, match="fof_halo_tag was found in multiple"):
        halos.evaluate(gas_count, scale=2)

    def gas_count(fof_halo_mass, rho, scale):
        return scale * len(rho)

    result = halos.evaluate(gas_count, format="numpy", scale=2)
    assert "gas_count" in result["halo_properties"].columns
    inserted = result["halo_properties"].select("gas_count").get_data("numpy")
    expected = halos.evaluate(gas_count, format="numpy", insert=False, scale=2)
    assert len(inserted) == len(halos)
    assert np.all(inserted == expected["gas_count"])
    assert np.all(inserted % 2 == 0)


def test_evaluate_inferred_with_datasets_resolves_ambiguity(halos):
    def mean_offset(fof_halo_center_x, x):
        return np.mean(x) - fof_halo_center_x

    with pytest.raises(ValueError, match="x was found in multiple datasets"):
        halos.evaluate(mean_offset)

    def explicit(halo_properties, dm_particles):
        return np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]

    narrowed = halos.with_datasets(["halo_properties", "dm_particles"])
    inferred = narrowed.evaluate(mean_offset, insert=False, format="numpy")
    expected = halos.evaluate(
        explicit,
        insert=False,
        format="numpy",
        halo_properties=["fof_halo_center_x"],
        dm_particles=["x"],
    )
    assert np.allclose(inferred["mean_offset"], expected["explicit"])


def test_evaluate_inferred_nested_columns(halo_paths, galaxy_paths):
    collection = oc.open(*halo_paths, *galaxy_paths).take(50)

    def galaxy_count(fof_halo_mass, gal_mass_star):
        return len(gal_mass_star)

    result = collection.evaluate(galaxy_count, format="numpy")
    counts = result["halo_properties"].select("galaxy_count").get_data("numpy")

    halo_tags = collection["halo_properties"].select("fof_halo_tag").get_data("numpy")
    galaxy_tags = (
        collection["galaxies"]["galaxy_properties"]
        .select("fof_halo_tag")
        .get_data("numpy")
    )
    expected = np.array([np.sum(galaxy_tags == tag) for tag in halo_tags])
    assert np.all(counts == expected)


def test_evaluate_inferred_linked_columns_only(halos):
    def agn_count(mbh):
        return len(mbh)

    counts = halos.evaluate(agn_count, format="numpy", insert=False)
    assert len(counts["agn_count"]) == len(halos)


def test_evaluate_without_columns_or_datasets_raises(halos):
    def constant():
        return 1

    with pytest.raises(ValueError, match="names of datasets or columns"):
        halos.evaluate(constant)

    def unknown(not_a_column):
        return 1

    with pytest.raises(ValueError, match="not_a_column"):
        halos.evaluate(unknown)
