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


# Column names are data. A new column named like an option parameter of the
# function that adds it must be added as a column, not swallowed as an option.
RESERVED_COLUMN_NAMES = ["mode", "descriptions", "allow_overwrite", "dataset"]


@pytest.fixture
def memory_dataset_state(test_data):
    from opencosmo.dataset.state import state_from_evaluated_data
    from opencosmo.header import read_header

    header = read_header(test_data.healpix_map)
    return state_from_evaluated_data({"v": np.arange(5.0)}, header)


@pytest.mark.parametrize("name", RESERVED_COLUMN_NAMES)
def test_dataset_add_new_columns_accepts_reserved_names(memory_dataset_state, name):
    from opencosmo.dataset import operations as dsops

    new = dsops.add_new_columns(memory_dataset_state, {name: np.arange(5.0) * 2})
    assert name in new.columns
    assert np.array_equal(
        dsops.get_data(dsops.select(new, name), "numpy", wrap_single=True)[name],
        np.arange(5.0) * 2,
    )


# `mode` is a public keyword of select, so it can never name a derived column there.
@pytest.mark.parametrize("name", ["descriptions", "allow_overwrite", "dataset"])
def test_dataset_select_accepts_reserved_derived_names(memory_dataset_state, name):
    from opencosmo.dataset import operations as dsops

    selected = dsops.select(memory_dataset_state, **{name: col("v") * 2})
    assert list(selected.columns) == [name]


@pytest.mark.parametrize("name", RESERVED_COLUMN_NAMES)
def test_structure_evaluate_inserts_reserved_output_names(halo_paths, name):
    import opencosmo as oc

    collection = oc.open(*halo_paths).take(5)

    def evaluated(dm_particles, halo_properties):
        return {name: float(np.mean(np.asarray(dm_particles["x"].value)))}

    result = collection.evaluate(
        evaluated,
        dm_particles=["x"],
        halo_properties=["fof_halo_center_x"],
        insert=True,
    )
    assert name in result["halo_properties"].columns


@pytest.fixture
def halo_paths(test_data):
    return test_data.snapshot.primary.halos


def _healpix_index(level: int):
    from opencosmo.spatial.index import HealpixIndex

    data = {}
    for lvl in range(level + 1):
        n = 12 * 4**lvl
        full = lvl == level
        data[f"level_{lvl}"] = {
            "start": np.arange(n, dtype=np.int64) if full else np.zeros(n, np.int64),
            "size": np.ones(n, np.int64) if full else np.zeros(n, np.int64),
        }
    return HealpixIndex(level, data)


def test_expand_lightcone_region_converts_cone_to_healpix():
    import opencosmo as oc
    from opencosmo.io.iopen import _expand_lightcone_region
    from opencosmo.spatial.region import HealpixRegion

    cone = oc.make_cone((45 * u.deg, 10 * u.deg), 20 * u.deg)
    expanded = _expand_lightcone_region(cone, _healpix_index(2))
    assert isinstance(expanded, HealpixRegion)
    assert expanded.nside == 4
    assert len(expanded.pixels) > 0


def test_expand_lightcone_region_keeps_healpix_region_pixels():
    from opencosmo.io.iopen import _expand_lightcone_region
    from opencosmo.spatial.region import HealpixRegion

    region = HealpixRegion(np.array([0, 1, 5], dtype=np.int64), 2)
    expanded = _expand_lightcone_region(region, _healpix_index(2))
    # nside 2 -> 4 multiplies each pixel by 4
    assert expanded.nside == 4
    assert set(expanded.pixels) == set(range(8)) | set(range(20, 24))


def test_expand_lightcone_region_rejects_3d_region():
    import opencosmo as oc
    from opencosmo.io.iopen import _expand_lightcone_region

    box = oc.make_box((0, 0, 0), (1, 1, 1))
    with pytest.raises(TypeError, match="2-dimensional"):
        _expand_lightcone_region(box, _healpix_index(2))
