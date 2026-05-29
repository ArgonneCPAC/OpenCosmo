from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, Optional
from warnings import warn
from weakref import finalize

import astropy.units as u
import numpy as np

from opencosmo.column.cache import ColumnCache
from opencosmo.column.column import RawColumn
from opencosmo.column.select import get_column_selection
from opencosmo.dataset.columns import add_columns, resort
from opencosmo.dataset.instantiate import instantiate_dataset
from opencosmo.dataset.output import get_derived_column_names, make_dataset_schema
from opencosmo.handler.empty import EmptyHandler
from opencosmo.handler.hdf5 import Hdf5Handler
from opencosmo.index import empty, into_array, mask, project, single_chunk
from opencosmo.plugins.contexts import (
    DatasetInstantiateCtx,
    HookPoint,
    IndexUpdateCtx,
    PostSortCtx,
)
from opencosmo.plugins.hook import fold
from opencosmo.units import UnitConvention
from opencosmo.units.handler import (
    make_unit_handler_from_hdf5,
    make_unit_handler_from_units,
)

if TYPE_CHECKING:
    from uuid import UUID

    from opencosmo.column.column import (
        ColumnMask,
        ConstructedColumn,
    )
    from opencosmo.handler.protocols import DataCache, DataHandler
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region
    from opencosmo.spatial.tree import Tree
    from opencosmo.units.handler import UnitHandler


def deregister_state(id: int, cache: DataCache):
    cache.deregister_column_group(id)


def sort_data(
    data: dict[str, np.ndarray], sort_by: tuple[str, bool] | None, state: DatasetState
):
    if sort_by is None:
        return data
    sort_column = data[sort_by[0]]
    order = np.argsort(sort_column)
    if sort_by[1]:
        order = order[::-1]

    data = {key: value[order] for key, value in data.items()}
    if sort_by[0] not in state.columns:
        data.pop(sort_by[0])
    return fold(HookPoint.PostSort, PostSortCtx(state, data, np.argsort(order))).data


@dataclass(frozen=True)
class DatasetState:
    """
    Main state container for the Dataset. This holds the full backing state of
    a dataset: data handler, cache, unit handler, header, region, spatial tree,
    and derived-column producers.

    The dataclass exposes only basic lookup properties. Manipulation is done
    via the standalone functions in this module; each returns a new
    ``DatasetState``. Sanitization of user-supplied arguments (wildcard
    expansion, format strings, scale-factor lookup, etc.) happens in the
    ``Dataset`` wrapper before reaching these functions.
    """

    producers: dict[UUID, ConstructedColumn]
    raw_data_handler: DataHandler
    cache: DataCache
    unit_handler: UnitHandler
    header: OpenCosmoHeader
    column_map: dict[str, UUID]
    region: Region
    open_kwargs: dict[str, Any]
    sort_key: Optional[tuple[str, bool]]
    metadata_columns: frozenset[str]
    tree: Optional[Tree] = None

    def __post_init__(self):
        self.cache.register_column_group(id(self), self.column_map)
        finalize(self, deregister_state, id(self), self.cache)

    # ------------------------------------------------------------------
    # Basic lookup properties
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        return [c for c in self.column_map if c not in self.metadata_columns]

    @property
    def meta_columns(self) -> list[str]:
        return [c for c in self.column_map if c in self.metadata_columns]

    @property
    def descriptions(self):
        all_descriptions = {}
        for producer in self.producers.values():
            update = {name: producer.description for name in producer.produces}
            all_descriptions |= update
        all_descriptions |= self.cache.descriptions

        return {
            name: description
            for name, description in all_descriptions.items()
            if name in self.columns
        }

    @property
    def kwargs(self):
        return self.open_kwargs

    @property
    def raw_index(self):
        if (si := get_sorted_index(self)) is not None:
            ni = into_array(self.raw_data_handler.index)
            return ni[si]
        return self.raw_data_handler.index

    @property
    def units(self):
        units = self.unit_handler.current_units
        return {name: units[name] for name in self.columns}

    @property
    def convention(self):
        return self.unit_handler.current_convention

    def __len__(state: DatasetState) -> int:
        if isinstance(state.raw_data_handler, EmptyHandler):
            return len(state.cache)
        return len(state.raw_data_handler)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Factory functions (replace classmethods)
# ---------------------------------------------------------------------------


def state_from_target(
    target: DatasetTarget,
    unit_convention: UnitConvention,
    region: Region,
    open_kwargs: dict[str, Any],
    index: Optional[DataIndex] = None,
    metadata_group: Optional[str] = None,
    tree: Optional[Tree] = None,
) -> DatasetState:
    data_group = target["dataset_group"]
    if "load" in data_group.keys():
        load_conditions = dict(data_group["load/if"].attrs)
    else:
        load_conditions = None

    handler = Hdf5Handler.from_columns(
        target["columns"],
        index,
        metadata_group,
        load_conditions,
    )
    unit_handler = make_unit_handler_from_hdf5(
        target["columns"], target["header"], unit_convention
    )
    meta_column_names = frozenset(
        col.name.split("/")[-1]
        for col in target["columns"]
        if metadata_group and col.name.split("/")[-2] == metadata_group
    )
    descriptions = handler.descriptions

    raw_producers = [
        RawColumn(cname, descriptions.get(cname, "None")) for cname in handler.columns
    ]
    column_map = {p.name: p.uuid for p in raw_producers}
    producers: dict[UUID, ConstructedColumn] = {p.uuid: p for p in raw_producers}
    cache = ColumnCache.empty()
    return DatasetState(
        producers=producers,
        raw_data_handler=handler,
        cache=cache,
        unit_handler=unit_handler,
        header=target["header"],
        column_map=column_map,
        region=region,
        open_kwargs=open_kwargs,
        sort_key=None,
        metadata_columns=meta_column_names,
        tree=tree,
    )


def state_in_memory(
    data_columns: dict,
    metadata_columns: dict,
    header: OpenCosmoHeader,
    unit_convention: UnitConvention,
    region: Region,
    open_kwargs: dict[str, Any],
    descriptions: Optional[dict[str, str]] = None,
    index: Optional[DataIndex] = None,
    tree: Optional[Tree] = None,
) -> DatasetState:
    descriptions = descriptions or {}

    all_columns = dict(data_columns) | dict(metadata_columns)
    raw_producers = [
        RawColumn(cname, descriptions.get(cname, "None"))
        for cname in all_columns.keys()
    ]
    column_map = {p.name: p.uuid for p in raw_producers}
    producers: dict[UUID, ConstructedColumn] = {p.uuid: p for p in raw_producers}

    cache = ColumnCache.empty()
    if all_columns:
        uuid_data = {p.uuid: {p.name: all_columns[p.name]} for p in raw_producers}
        cache.add_data(uuid_data, descriptions)

    units: dict[str, u.Unit] = {}
    for name, column in all_columns.items():
        units[name] = None
        if isinstance(column, u.Quantity):
            units[name] = column.unit

    unit_handler = make_unit_handler_from_units(units, header, unit_convention)

    return DatasetState(
        producers=producers,
        raw_data_handler=EmptyHandler(),
        cache=cache,
        unit_handler=unit_handler,
        header=header,
        column_map=column_map,
        region=region,
        open_kwargs=open_kwargs,
        sort_key=None,
        metadata_columns=frozenset(metadata_columns.keys()),
        tree=tree,
    )


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def exit_state(state: DatasetState, *exec_details):
    return None


def get_data(
    state: DatasetState,
    format: str = "astropy",
    unpack: bool = True,
    metadata_columns: list = [],
    wrap_single: bool = False,
    ignore_sort: bool = False,
):
    """
    Materialize the data for ``state`` in the requested format.

    Reads through the cache, applies pending derived/evaluated columns,
    looks up the appropriate scale factor (in physical convention), and
    converts the result into ``format``.
    """
    from opencosmo.dataset.formats import convert_data
    from opencosmo.units.converters import get_scale_factor

    if state.convention.value == "physical":
        scale_factor = get_scale_factor(
            state, state.header.cosmology, state.header.file.redshift
        )
        unit_kwargs = {"scale_factor": scale_factor}
    else:
        unit_kwargs = {}

    data = _get_raw_data(
        state,
        ignore_sort=ignore_sort,
        metadata_columns=metadata_columns,
        unit_kwargs=unit_kwargs,
    )
    if unpack:
        data = {
            key: value[0]
            if isinstance(value, np.ndarray) and len(value) == 1
            else value
            for key, value in data.items()
        }

    return convert_data(data, format, wrap_single=wrap_single)


def _get_raw_data(
    state: DatasetState,
    ignore_sort: bool = False,
    metadata_columns: list = [],
    unit_kwargs: dict = {},
) -> dict:
    """
    Read raw data for ``state`` as a {name: numpy-or-Quantity} dict, with units
    applied and (optionally) sorting applied. Used internally by ``get_data``
    and by helpers that need pre-format-conversion data.
    """
    state = fold(HookPoint.DatasetInstantiate, DatasetInstantiateCtx(state)).state
    data = instantiate_dataset(
        list(state.producers.values()),
        state.column_map,
        state.raw_data_handler,
        state.cache,
        state.unit_handler,
        unit_kwargs,
        None if (ignore_sort or state.sort_key is None) else state.sort_key[0],
    )

    if missing := set(state.columns).difference(data.keys()):
        raise RuntimeError(
            f"Some columns are missing from the output! This is likely a bug. Please report it on GitHub. Missing: {missing}"
        )

    if not ignore_sort:
        data = sort_data(data, state.sort_key, state)

    new_order = list(state.columns)
    for name in metadata_columns:
        if name in state.metadata_columns:
            new_order.append(name)

    return {name: data[name] for name in new_order}


def iter_rows(
    state: DatasetState,
    metadata_columns: list = [],
) -> Generator:
    """
    Iterate over the rows of a given DatasetState. The appropriate scale factor
    (in physical convention) is applied so the yielded values match what
    ``get_data`` would return.
    """
    from opencosmo.units.converters import get_scale_factor

    if state.convention.value == "physical":
        scale_factor = get_scale_factor(
            state, state.header.cosmology, state.header.file.redshift
        )
        unit_kwargs = {"scale_factor": scale_factor}
    else:
        unit_kwargs = {}

    derived_to_collect = (
        set(state.columns)
        .difference(state.cache.columns)
        .difference(state.raw_data_handler.columns)
    )
    derived_storage: dict[str, list[np.ndarray]] = {
        name: [] for name in derived_to_collect
    }
    total_length = len(state)
    chunk_ranges = [
        (i, min(i + 1000, total_length)) for i in range(0, total_length, 1000)
    ]
    if not chunk_ranges:
        raise StopIteration

    try:
        for start, end in chunk_ranges:
            chunk = take_rows(state, single_chunk(start, end - start))
            data = _get_raw_data(
                chunk, metadata_columns=metadata_columns, unit_kwargs=unit_kwargs
            )
            for name in derived_to_collect:
                derived_storage[name].append(data[name])

            for i in range(len(chunk)):
                yield {name: column[i] for name, column in data.items()}
        all_derived = {
            name: np.concatenate(arr) for name, arr in derived_storage.items()
        }
        derived_storage = resort(all_derived, get_sorted_index(state))
        if derived_storage:
            uuid_keyed: dict = {}
            for name, arr in derived_storage.items():
                uuid = state.column_map[name]
                uuid_keyed.setdefault(uuid, {})[name] = arr
            state.cache.add_data(uuid_keyed, {})
    except GeneratorExit:
        pass
    except BaseException:
        raise


def get_metadata(
    state: DatasetState, columns: list = [], ignore_sort: bool = False
) -> dict:
    names = list(columns) if columns else list(state.metadata_columns)
    data = instantiate_dataset(
        list(state.producers.values()),
        {name: state.column_map[name] for name in names},
        state.raw_data_handler,
        state.cache,
        state.unit_handler,
        {},
        None,
    )
    if ignore_sort:
        return data

    sorted_index = get_sorted_index(state)
    if sorted_index is not None:
        data = {name: values[sorted_index] for name, values in data.items()}
    return data


def make_schema(state: DatasetState, name: Optional[str] = None) -> Schema:
    """
    Build the write-time schema for this state. Includes the data columns, the
    spatial-index tree (if present) filtered to match the current index, and
    the header dump.
    """
    producers = list(state.producers.values())
    columns = set(state.column_map.keys()).difference(state.metadata_columns)
    derived_names = get_derived_column_names(producers, columns)
    if derived_names:
        selected = select(state, derived_names)
        converted = with_units(selected, state.unit_handler.base_convention)
        derived_data = _get_raw_data(converted, ignore_sort=True)
    else:
        derived_data = {}
    schema = make_dataset_schema(
        producers,
        state.raw_data_handler,
        state.cache,
        state.column_map,
        state.meta_columns,
        state.header,
        state.region,
        derived_data,
        name,
    )

    if state.tree is not None:
        tree = state.tree.apply_index(state.raw_index)
        schema.children["index"] = tree.make_schema()
    schema.children["header"] = state.header.dump()
    return schema


def with_new_columns(
    state: DatasetState,
    descriptions: dict[str, str] = {},
    allow_overwrite: bool = False,
    **new_columns: ConstructedColumn | np.ndarray | u.Quantity,
) -> DatasetState:
    """
    Add columns to a given state
    """
    new_producers_list, new_column_map, new_unit_handler = add_columns(
        list(state.producers.values()),
        state.unit_handler,
        state.cache,
        state.column_map,
        get_sorted_index(state),
        descriptions,
        new_columns,
        len(state),
        allow_overwrite=allow_overwrite,
    )
    return dataclasses.replace(
        state,
        producers={p.uuid: p for p in new_producers_list},
        column_map=new_column_map,
        unit_handler=new_unit_handler,
    )


def with_region(state: DatasetState, region: Region) -> DatasetState:
    return dataclasses.replace(state, region=region)


def select(state: DatasetState, columns: set[str], drop: bool = False) -> DatasetState:
    """
    Select a set of columns
    """
    selections, missing = get_column_selection(state.columns, columns)
    if missing:
        raise ValueError(
            f"Columns are included that are not in this dataset: {missing}"
        )
    elif not selections and columns:
        raise ValueError("No columns matched the provided wildcards!")

    if drop:
        selections = set(state.columns) - selections

    new_column_map = {n: state.column_map[n] for n in selections}
    new_column_map |= {n: state.column_map[n] for n in state.metadata_columns}
    return dataclasses.replace(state, column_map=new_column_map)


def sort_by(
    state: DatasetState, column_name: Optional[str], invert: bool
) -> DatasetState:
    if column_name is None:
        sort_key = None
    elif column_name not in state.columns:
        raise ValueError(f"This dataset has no column {column_name}")
    else:
        sort_key = (column_name, invert)

    return dataclasses.replace(state, sort_key=sort_key)


def get_sorted_index(state: DatasetState) -> np.ndarray | None:
    if state.sort_key is not None:
        column = _get_raw_data(select(state, {state.sort_key[0]}), ignore_sort=True)[
            state.sort_key[0]
        ]
        sorted_idx = np.argsort(column)
        if state.sort_key[1]:
            sorted_idx = sorted_idx[::-1]
    else:
        sorted_idx = None

    return sorted_idx


def take_rows(state: DatasetState, rows: DataIndex) -> DatasetState:
    """
    Take a set of rows. The associated "take" functions in the
    dataset all delegate to this function.
    """
    if len(state) == 0:
        return state
    rows = fold(HookPoint.IndexUpdate, IndexUpdateCtx(state, rows)).index
    sorted_idx = get_sorted_index(state)
    if sorted_idx is not None:
        rows = np.sort(sorted_idx[into_array(rows)])
    new_handler = state.raw_data_handler.take(rows)
    new_cache = state.cache.take(rows)
    return dataclasses.replace(state, raw_data_handler=new_handler, cache=new_cache)


def with_units(
    state: DatasetState,
    convention: Optional[str] = None,
    conversions: dict[u.Unit, u.Unit] = {},
    columns: dict[str, u.Unit] = {},
) -> DatasetState:
    """
    Update the units of a given state. If ``convention`` is set, the header
    is also updated to reflect the new convention.
    """
    if convention is None:
        convention_ = state.unit_handler.current_convention
    else:
        convention_ = UnitConvention(convention)

    if (
        convention_ == UnitConvention.SCALEFREE
        and UnitConvention(state.header.file.unit_convention)
        != UnitConvention.SCALEFREE
    ):
        raise ValueError(
            f"Cannot convert units with convention {state.header.file.unit_convention} to convention scalefree"
        )
    column_keys = set(columns.keys())
    missing_columns = column_keys - set(state.columns)
    if missing_columns:
        raise ValueError(f"Dataset does not have columns {missing_columns}")

    new_handler = state.unit_handler.with_convention(convention_).with_conversions(
        conversions, columns
    )

    if convention_ == state.unit_handler.current_convention:
        cache = state.cache.create_child()
    else:
        all_derived_names: set[str] = set()
        all_derived_names = reduce(
            lambda acc, col: acc.union(
                col.produces if not isinstance(col, RawColumn) else set()
            ),
            state.producers.values(),
            all_derived_names,
        ).intersection(state.columns)
        columns_to_drop = all_derived_names.union(state.raw_data_handler.columns)
        cache = state.cache.drop(columns_to_drop)

    new_header = state.header
    if convention is not None:
        new_header = state.header.with_units(convention)
    return dataclasses.replace(
        state, unit_handler=new_handler, cache=cache, header=new_header
    )


# ---------------------------------------------------------------------------
# Composite state operations (compose multiple primitives, no user-input
# sanitization). Callers normalize their inputs before invoking these.
# ---------------------------------------------------------------------------


def take(
    state: DatasetState,
    n: int,
    at: str = "random",
    mode: Literal["local", "global"] = "local",
) -> DatasetState:
    from opencosmo.dataset.take import (
        get_end_take_index,
        get_random_take_index,
    )

    if at == "start":
        return take_range(state, 0, n, mode)
    elif at == "end":
        take_index = get_end_take_index(n, state, state.sort_key, mode)
        return take_rows(state, take_index)
    elif at != "random":
        raise ValueError(f"Unknown take type {at}")

    row_indices = get_random_take_index(n, len(state), mode)
    return take_rows(state, row_indices)


def take_range(
    state: DatasetState,
    start: int,
    end: int,
    mode: Literal["local", "global"] = "local",
) -> DatasetState:
    from opencosmo.dataset.take import get_range_take_index

    if start < 0 or end < 0:
        raise ValueError("start and end must be positive.")
    if end < start:
        raise ValueError("end must be greater than start.")

    take_index = get_range_take_index(state, state.sort_key, start, end - start, mode)
    return take_rows(state, take_index)


def filter(state: DatasetState, *masks: ColumnMask) -> DatasetState:
    if not masks:
        return state
    bool_mask = np.ones(len(state), dtype=bool)
    for m in masks:
        bool_mask &= m.apply(state)
    return take_rows(state, np.where(bool_mask)[0])


def bound(
    state: DatasetState,
    region: Region,
    select_by: Optional[str] = None,
) -> DatasetState:
    from opencosmo.spatial import check

    if state.tree is None:
        raise AttributeError(
            "Your dataset does not contain a spatial index, "
            "so spatial querying is not available"
        )

    if not state.header.file.is_lightcone:
        columns = check.find_coordinates_3d(state, str(state.header.file.data_type))
        check_region = region.into_base_convention(
            state.unit_handler,  # type: ignore[arg-type]
            columns,
            state.convention,
            {
                "scale_factor": state.header.cosmology.scale_factor(
                    state.header.file.redshift
                ).value
            },
        )
    else:
        check_region = region

    if not state.region.intersects(check_region):
        return take_rows(state, empty())

    if not state.region.contains(check_region):
        warn(
            "You're querying with a region that is not fully contained by the "
            "region this dataset is in. This may result in unexpected behavior"
        )

    contained_index: DataIndex
    intersects_index: DataIndex
    contained_index, intersects_index = state.tree.query(check_region)

    contained_index = project(state.raw_index, contained_index)
    intersects_index = project(state.raw_index, intersects_index)

    check_state = take_rows(state, intersects_index)
    if not state.header.file.is_lightcone:
        check_state = with_units(check_state, "scalefree")

    if len(check_state) > 0:
        index_mask = check.check_containment(
            check_state, check_region, state.header.file
        )
        new_intersects_index = mask(intersects_index, index_mask)
    else:
        new_intersects_index = np.array([], dtype=np.int64)

    new_index = np.sort(
        np.concatenate([into_array(contained_index), into_array(new_intersects_index)])
    )

    return with_region(take_rows(state, new_index), check_region)


def evaluate(
    state: DatasetState,
    func: Callable,
    vectorize: bool = False,
    insert: bool = True,
    format: str = "astropy",
    batch_size: int = -1,
    allow_overwrite: bool = False,
    **evaluate_kwargs,
):
    from opencosmo.dataset.evaluate import build_evaluated_column, visit_dataset

    evaluated_column = build_evaluated_column(
        state, func, vectorize, insert, format, batch_size, evaluate_kwargs
    )

    if not insert:
        return visit_dataset(evaluated_column, state, batch_size)

    return with_new_columns(
        state,
        {},
        allow_overwrite,
        **{func.__name__: evaluated_column},
    )
