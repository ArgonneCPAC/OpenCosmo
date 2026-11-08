"""
Tests for StructureCollection.evaluate with column names as function arguments.
"""

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
