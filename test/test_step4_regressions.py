from __future__ import annotations

from types import SimpleNamespace

import astropy.units as u
import numpy as np

# plotting.py imports opencosmo.analysis.statistics; preload it to avoid a
# pre-existing package-init cycle when plotting is imported first.
import opencosmo.analysis.statistics  # noqa: F401
import pytest
from opencosmo.analysis.default_plotting_params import default_params
from opencosmo.analysis.plotting import _set_params
from opencosmo.collection.structure.structure import StructureCollection
from opencosmo.column.column import col
from opencosmo.handler.memory import InMemoryHandler


def test_structure_collection_dtype_reads_state_dtype():
    state = SimpleNamespace(dtype="halo_properties")
    collection = StructureCollection.__new__(StructureCollection)
    object.__setattr__(collection, "_StructureCollection__state", state)
    assert collection.dtype == "halo_properties"


def test_compound_mask_requires_names_covers_both_sides():
    mask = (col("a") > 1) & (col("b") < 2)
    assert mask.requires_names == {"a", "b"}
    assert not hasattr(mask, "requires")


def test_in_memory_sorted_take_maps_logical_rows_to_physical_rows():
    data = {"x": np.arange(5, dtype=np.int64)}
    handler = InMemoryHandler(
        data, {"x": None}, {"x": ""}, np.arange(5, dtype=np.int64)
    )
    sorted_index = np.array([4, 3, 2, 1, 0], dtype=np.int64)
    taken = handler.take(np.array([0, 2], dtype=np.int64), sorted_index)
    assert np.array_equal(taken.index, np.array([2, 4], dtype=np.int64))


def test_in_memory_sorted_take_accepts_chunked_index():
    data = {"x": np.arange(5, dtype=np.int64)}
    handler = InMemoryHandler(
        data, {"x": None}, {"x": ""}, np.arange(5, dtype=np.int64)
    )
    sorted_index = np.array([4, 3, 2, 1, 0], dtype=np.int64)
    # logical rows [0, 1] -> physical [4, 3] -> sorted [3, 4]
    chunked = (np.array([0], dtype=np.int64), np.array([2], dtype=np.int64))
    taken = handler.take(chunked, sorted_index)
    assert np.array_equal(taken.index, np.array([3, 4], dtype=np.int64))


def test_set_params_does_not_leak_unit_label_into_defaults():
    column = "fof_halo_mass"
    original_plotting = dict(default_params[column]["plotting"])

    labelled = _set_params(SimpleNamespace(units={column: u.Msun}), column)
    assert labelled["plotting"]["unit_label"]

    unlabelled = _set_params(SimpleNamespace(units={column: None}), column)
    assert unlabelled["plotting"]["unit_label"] is None

    assert default_params[column]["plotting"] == original_plotting
    assert "unit_label" not in default_params[column]["plotting"]


def test_set_params_fallback_for_unknown_column_is_fresh_each_call():
    ds = SimpleNamespace(units={"not_a_column": u.Msun})
    first = _set_params(ds, "not_a_column")
    second = _set_params(ds, "not_a_column")
    assert first is not second
    assert first["plotting"] is not second["plotting"]
    assert "not_a_column" not in default_params


def test_function_name_returns_def_name():
    from opencosmo.column.evaluate import function_name

    def offset_mass(x):
        return x

    assert function_name(offset_mass) == "offset_mass"


def test_function_name_rejects_lambda():
    from opencosmo.column.evaluate import function_name

    with pytest.raises(TypeError, match="Lambda"):
        function_name(lambda x: x)


def test_function_name_rejects_partial():
    from functools import partial

    from opencosmo.column.evaluate import function_name

    def scaled(x, factor):
        return x * factor

    with pytest.raises(TypeError, match="partial"):
        function_name(partial(scaled, factor=2))


def test_function_name_rejects_nameless_callable():
    from opencosmo.column.evaluate import function_name

    class Callable_:
        def __call__(self, x):
            return x

    with pytest.raises(TypeError, match="__name__"):
        function_name(Callable_())


def test_access_table_requires_one_cosmology():
    from opencosmo.header import get_access_table
    from opencosmo.units import UnitConvention

    with pytest.raises(ValueError, match="exactly one cosmology"):
        get_access_table([], UnitConvention.COMOVING, None)
