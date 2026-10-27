from __future__ import annotations

import dataclasses
from collections import defaultdict
from copy import copy
from functools import reduce
from typing import TYPE_CHECKING
from warnings import warn

import astropy.units as u
import numpy as np

from opencosmo.column.column import EvaluatedColumn, RawColumn
from opencosmo.column.reducer import default_reducer
from opencosmo.column.select import MissingColumnError, get_column_selection
from opencosmo.dataset import state as st
from opencosmo.dataset.columns import add_columns, resort
from opencosmo.dataset.evaluate import build_evaluated_column, visit_dataset
from opencosmo.dataset.formats import convert_data, verify_format
from opencosmo.dataset.graph import get_all_required_pairs
from opencosmo.dataset.instantiate import instantiate_dataset
from opencosmo.dataset.output import get_derived_column_names, make_dataset_schema
from opencosmo.dataset.take import (
    get_end_take_index,
    get_random_take_index,
    get_range_take_index,
)
from opencosmo.index import (
    empty,
    get_range,
    into_array,
    mask,
    project,
    reindex_column,
    single_chunk,
)
from opencosmo.mpi import gather_index, get_comm_world
from opencosmo.plugins.contexts import (
    DatasetInstantiateCtx,
    HookPoint,
    IndexUpdateCtx,
    PostSortCtx,
)
from opencosmo.plugins.hook import fold
from opencosmo.spatial import check
from opencosmo.units.convention import UnitConvention
from opencosmo.units.converters import get_scale_factor
from opencosmo.uuid import get_string_uuid

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Literal

    from opencosmo.column.column import (
        ColumnMask,
        CompoundColumnMask,
        ConstructedColumn,
        DerivedScalarValue,
    )
    from opencosmo.dataset.state import DatasetState
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema


def get_data(
    state: DatasetState,
    format: str,
    unpack: bool = False,
    wrap_single: bool = False,
    **kwargs,
):
    verify_format(format)

    if state.convention.value == "physical":
        scale_factor = get_scale_factor(
            state, state.header.cosmology, state.header.file.redshift
        )
        unit_kwargs = {"scale_factor": scale_factor}
    else:
        unit_kwargs = {}

    data = __get_data(
        state,
        unit_kwargs=unit_kwargs,
        **kwargs,
    )  # dict
    if unpack:
        data = {
            key: value[0]
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == 1
            else value
            for key, value in data.items()
        }

    return convert_data(data, format, wrap_single=wrap_single)


def filter(
    state: DatasetState, *masks: ColumnMask | CompoundColumnMask, mode: str = "global"
):
    reducer = default_reducer(mode)
    masks = tuple(m.with_reducer(reducer) for m in masks)
    bool_mask = np.ones(len(state), dtype=bool)
    required_columns: set[str] = set()
    for m in masks:
        required_columns |= m.requires_names

    selected_state = __select(state, required_columns)
    data = get_data(selected_state, "astropy", wrap_single=True)

    for m in masks:
        bool_mask &= m.apply(data)

    new_state = __take_rows(state, np.where(bool_mask)[0])
    return new_state


def select(
    state: DatasetState,
    *columns: str | Iterable[str],
    mode: str = "global",
    **derived_columns: ConstructedColumn | DerivedScalarValue,
):
    from opencosmo.column.column import Column, DerivedScalarValue
    from opencosmo.column.reducer import default_reducer

    all_columns: set[str] = set()
    for col_group in columns:
        if isinstance(col_group, str):
            col_group = {col_group}
        all_columns.update(col_group)

    scalars: dict[str, DerivedScalarValue] = {}
    non_scalars: dict[str, ConstructedColumn] = {}
    for name, col in derived_columns.items():
        if isinstance(col, DerivedScalarValue):
            scalars[name] = col
        else:
            non_scalars[name] = col

    if scalars and (all_columns or non_scalars):
        raise ValueError(
            "Scalar selections cannot be mixed with column selections. "
            "Call select() with only scalar kwargs, or only column selections."
        )

    reducer = default_reducer(mode)
    derived_columns = {
        k: v.with_reducer(reducer) if isinstance(v, (Column, DerivedScalarValue)) else v
        for k, v in derived_columns.items()
    }

    new_state = state
    if derived_columns:
        new_state = __with_new_columns(new_state, {}, False, **derived_columns)
        all_columns.update(derived_columns.keys())

    if all_columns:
        new_state = __select(new_state, all_columns)
    return new_state


def drop(state: DatasetState, *columns: str | Iterable[str]):
    all_columns: set[str] = set()
    for col_group in columns:
        if isinstance(col_group, str):
            col_group = {col_group}
        all_columns.update(col_group)
    return __select(state, all_columns, drop=True)


def sort_by(state: DatasetState, column: str | None, invert: bool) -> DatasetState:
    if column is None:
        sort_key = None
    elif column not in state.columns:
        raise ValueError(f"This dataset has no column {column}")
    else:
        sort_key = (column, invert, False)

    return dataclasses.replace(state, sort_key=sort_key)


def with_new_columns(
    state: DatasetState,
    descriptions: str | dict[str, str] | None = None,
    allow_overwrite: bool = False,
    mode: str = "global",
    **new_columns: ConstructedColumn | np.ndarray | u.Quantity,
):
    from opencosmo.column.column import Column, DerivedScalarValue
    from opencosmo.column.reducer import default_reducer

    if descriptions is None:
        descriptions = {}
    if any(isinstance(col, DerivedScalarValue) for col in new_columns.values()):
        raise ValueError(
            "Scalar values cannot be added to an existing dataset, but can be retrieved with Dataset.select()"
        )
    reducer = default_reducer(mode)
    new_columns = {
        k: v.with_reducer(reducer) if isinstance(v, Column) else v
        for k, v in new_columns.items()
    }
    if isinstance(descriptions, str):
        descriptions = {key: descriptions for key in new_columns}
    return __with_new_columns(state, descriptions, allow_overwrite, **new_columns)


def evaluate(
    state: DatasetState,
    func: Callable,
    vectorize=False,
    insert=True,
    format="astropy",
    batch_size: int = -1,
    allow_overwrite: bool = False,
    require_length: bool = True,
    **evaluate_kwargs,
):
    verify_format(format)
    evaluated_column = build_evaluated_column(
        state,
        func,
        vectorize,
        insert,
        format,
        batch_size,
        evaluate_kwargs,
    )

    if not insert:
        output = visit_dataset(evaluated_column, state, batch_size)
        return output
    return with_new_columns(
        state, allow_overwrite=allow_overwrite, **{func.__name__: evaluated_column}
    )


def evaluate_to_dataset(
    state: DatasetState,
    func: Callable,
    vectorize: bool = False,
    format: str = "astropy",
    batch_size: int = -1,
    allow_overwrite: bool = False,
    **evaluate_kwargs,
) -> DatasetState:
    baseline_state = with_units(state, state.unit_handler.base_convention.value, {})
    result: dict[str, np.ndarray | u.Quantity] = evaluate(
        baseline_state,
        func,
        vectorize,
        False,
        format,
        batch_size,
        allow_overwrite,
        **evaluate_kwargs,
    )

    output_length: int | None = None
    for name, output in result.items():
        if not isinstance(output, (np.ndarray, u.Quantity)):
            raise TypeError(
                f"Evaluate output {name!r} must be a NumPy array or Astropy quantity, "
                f"not {type(output).__name__}"
            )
        if output.ndim == 0:
            raise TypeError(
                f"Evaluate output {name!r} must be a NumPy array or Astropy quantity "
                "with a length"
            )
        if output_length is None:
            output_length = len(output)
        elif len(output) != output_length:
            raise ValueError("Evaluate output columns must have equal lengths")

    evaluated = st.state_from_evaluated_data(result, baseline_state.header)
    return with_units(
        evaluated,
        state.unit_handler.current_convention.value,
        state.unit_handler.blanket_conversions,
    )


def take_range(
    state: DatasetState,
    start: int,
    end: int,
    mode: Literal["local", "global"],
) -> DatasetState:
    if start < 0 or end < 0:
        raise ValueError("start and end must be positive.")
    if end < start:
        raise ValueError("end must be greater than start.")

    take_index = get_range_take_index(state, state.sort_key, start, end - start, mode)
    return __take_rows(state, take_index)


def take_rows(state: DatasetState, rows: DataIndex) -> DatasetState:
    row_range = get_range(rows)
    if row_range[0] < 0 or row_range[1] > len(state):
        raise ValueError(
            "Row indices must be between 0 and the length of this dataset - 1!"
        )

    return __take_rows(state, rows)


def take(
    state: DatasetState,
    n: int,
    at: str,
    mode: Literal["local", "global"],
) -> DatasetState:
    if at == "start":
        return take_range(state, 0, n, mode)
    elif at == "end":
        take_index = get_end_take_index(n, state, state.sort_key, mode)
        return take_rows(state, take_index)
    elif at != "random":
        raise ValueError(f"Unknown take type {at}")

    row_indices = get_random_take_index(n, len(state), mode)
    return take_rows(state, row_indices)


def rows(
    state: DatasetState,
    include_units: bool = True,
):
    if state.convention.value == "physical":
        scale_factor = get_scale_factor(
            state, state.header.cosmology, state.header.file.redshift
        )
        unit_kwargs = {"scale_factor": scale_factor}
    else:
        unit_kwargs = {}

    for row in __iter_rows(state, unit_kwargs):
        output_data = row
        if not isinstance(output_data, dict):
            output_data = {state.columns[0]: row}

        if not include_units:
            output_data = {
                name: val.value if isinstance(val, u.Quantity) else val
                for name, val in output_data.items()
            }
        yield output_data


def bound(state: DatasetState, region, select_by):
    if state.spatial_index is None:
        raise AttributeError(
            "Your dataset does not contain a spatial index, "
            "so spatial querying is not available"
        )

    if not state.header.file.is_lightcone:
        columns = check.find_coordinates_3d(
            state, str(state.header.file.data_type), select_by
        )

        check_region = region.regularize(
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

    if state.region is None:
        raise RuntimeError("A spatially indexed dataset must have a region")
    if not state.region.intersects(check_region):
        return __take_rows(state, empty())

    if not state.region.contains(check_region):
        warn(
            "You're querying with a region that is not fully contained by the "
            "region this dataset is in. This may result in unexpected behavior"
        )

    contained_index: DataIndex
    intersects_index: DataIndex
    contained_index, intersects_index = state.spatial_index.query(check_region)
    contained_index = project(state.raw_index, contained_index)
    intersects_index = project(state.raw_index, intersects_index)

    check_state = __take_rows(state, intersects_index)
    if not state.header.file.is_lightcone:
        check_state = with_units(check_state, "scalefree", {})

    if len(check_state) > 0:
        index_mask = check.check_containment(
            check_state, check_region, state.header.file, select_by
        )
        new_intersects_index = mask(intersects_index, index_mask)
    else:
        new_intersects_index = np.array([], dtype=np.int64)

    new_index = np.sort(
        np.concatenate([into_array(contained_index), into_array(new_intersects_index)])
    )

    new_state = __take_rows(state, new_index)
    return dataclasses.replace(new_state, region=check_region)


def with_units(
    state: DatasetState,
    convention: str | None,
    conversions: dict[u.Unit, u.Unit],
    **columns: u.Unit,
) -> DatasetState:
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
    return __with_units(
        state,
        convention_,
        conversions,
        columns,
        state.header.cosmology,
        state.header.file.redshift,
    )


def __sort_data(
    data: dict[str, np.ndarray],
    sort_by: tuple[str, bool, bool] | None,
    state: DatasetState,
) -> dict[str, np.ndarray]:
    if sort_by is None:
        return data
    sort_column = data[sort_by[0]]
    order = np.argsort(sort_column)
    if sort_by[1]:
        order = order[::-1]

    data = {key: value[order] for key, value in data.items()}
    if sort_by[2] and set(data) != set(sort_by[0]):
        data.pop(sort_by[0])
    return fold(HookPoint.PostSort, PostSortCtx(state, data, np.argsort(order))).data


def __get_data(
    state: DatasetState,
    ignore_sort: bool = False,
    unit_kwargs: dict | None = None,
) -> dict:
    if unit_kwargs is None:
        unit_kwargs = {}
    state = fold(HookPoint.DatasetInstantiate, DatasetInstantiateCtx(state)).state
    data = instantiate_dataset(
        state.producers,
        state.column_map,
        state.raw_data_handler,
        state.cache,
        state.unit_handler,
        unit_kwargs,
        None if (ignore_sort or state.sort_key is None) else state.sort_key[0],
    )
    if missing := set(state.columns).difference(data):
        raise RuntimeError(
            "Some columns are missing from the output! This is likely a bug. "
            f"Please report it on GitHub. Missing: {missing}"
        )
    if not ignore_sort:
        data = __sort_data(data, state.sort_key, state)

    new_order = list(state.columns)
    if state.sort_key is not None and not new_order:
        new_order = [state.sort_key[0]]
    return {name: data[name] for name in new_order}


def __iter_rows(state: DatasetState, unit_kwargs: dict | None = None):
    if unit_kwargs is None:
        unit_kwargs = {}
    derived_to_collect = (
        set(state.columns)
        .difference(state.cache.columns)
        .difference(state.raw_data_handler.columns)
    )
    derived_storage: dict[str, list[np.ndarray]] = {
        name: [] for name in derived_to_collect
    }
    chunk_ranges = [(i, min(i + 1000, len(state))) for i in range(0, len(state), 1000)]
    if not chunk_ranges:
        return

    try:
        for start, end in chunk_ranges:
            chunk = __take_rows(state, single_chunk(start, end - start))
            data = __get_data(chunk, unit_kwargs=unit_kwargs)
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
                uuid_keyed.setdefault(state.column_map[name], {})[name] = arr
            state.cache.add_data(uuid_keyed, {})
    except GeneratorExit:
        pass


def make_schema(state: DatasetState, path: str) -> Schema:
    producers = list(state.producers.values())
    derived_names = get_derived_column_names(producers, set(state.column_map))
    if derived_names:
        selected = __select(state, derived_names)
        converted = __with_units(
            selected, state.unit_handler.base_convention, {}, {}, None, None
        )
        derived_data = __get_data(converted, ignore_sort=True)
    else:
        derived_data = {}

    column_map = copy(state.column_map)
    if state.sort_key is not None and state.sort_key[2]:
        column_map.pop(state.sort_key[0])
    return make_dataset_schema(
        producers,
        state.raw_data_handler,
        state.cache,
        column_map,
        state.header,
        state.spatial_index,
        state.region,
        state.raw_index,
        derived_data,
        get_string_uuid(path),
        path.split("/")[-1],
    )


def __with_new_columns(
    state: DatasetState,
    descriptions: dict[str, str] | None = None,
    allow_overwrite: bool = False,
    **new_columns: ConstructedColumn | np.ndarray | u.Quantity,
) -> DatasetState:
    if descriptions is None:
        descriptions = {}
    producers, column_map, unit_handler = add_columns(
        state.producers,
        state.unit_handler,
        state.cache,
        state.column_map,
        set(state.producers),
        get_sorted_index(state),
        descriptions,
        new_columns,
        len(state),
        allow_overwrite=allow_overwrite,
    )
    return dataclasses.replace(
        state, producers=producers, column_map=column_map, unit_handler=unit_handler
    )


def __select(
    state: DatasetState, columns: set[str], drop: bool = False
) -> DatasetState:
    selections, missing = get_column_selection(state.columns, columns)
    if (
        len(columns) > 1
        and state.sort_key is not None
        and state.sort_key[2]
        and state.sort_key[0] in columns
    ):
        missing.add(state.sort_key[0])
    elif (
        len(columns) == 1
        and state.sort_key is not None
        and missing == {state.sort_key[0]}
    ):
        selections, missing = columns, set()
    if missing:
        raise MissingColumnError(
            f"Columns are included that are not in this dataset: {missing}"
        )
    if not selections and columns:
        raise MissingColumnError("No columns matched the provided wildcards!")
    if drop:
        selections = set(state.columns) - selections

    sort_key = state.sort_key
    if sort_key is not None and sort_key[0] not in selections:
        selections.add(sort_key[0])
        sort_key = (sort_key[0], sort_key[1], True)
    return dataclasses.replace(
        state,
        column_map={name: state.column_map[name] for name in selections},
        sort_key=sort_key,
    )


def get_sorted_index(state: DatasetState) -> np.ndarray | None:
    if state.sort_key is None:
        return None
    column = __get_data(__select(state, {state.sort_key[0]}), ignore_sort=True)[
        state.sort_key[0]
    ]
    sorted_idx = np.argsort(column)
    return sorted_idx[::-1] if state.sort_key[1] else sorted_idx


def __take_rows(state: DatasetState, rows: DataIndex) -> DatasetState:
    if len(state) == 0:
        return state
    rows = fold(HookPoint.IndexUpdate, IndexUpdateCtx(state, rows)).index
    if (sorted_idx := get_sorted_index(state)) is not None:
        rows = np.sort(sorted_idx[into_array(rows)])
    return dataclasses.replace(
        state,
        raw_data_handler=state.raw_data_handler.take(rows),
        cache=state.cache.take(rows),
    )


def __with_units(
    state: DatasetState,
    convention: UnitConvention,
    conversions: dict[u.Unit, u.Unit],
    columns: dict[str, u.Unit],
    cosmology,
    redshift,
) -> DatasetState:
    unit_handler = state.unit_handler.with_convention(convention).with_conversions(
        conversions, columns
    )
    if convention == state.unit_handler.current_convention:
        cache = state.cache.create_child()
    else:
        derived_names: set[str] = reduce(
            lambda names, column: names.union(
                column.produces if not isinstance(column, RawColumn) else set()
            ),
            state.producers.values(),
            set[str](),
        ).intersection(state.columns)
        cache = state.cache.drop(derived_names.union(state.raw_data_handler.columns))
    return dataclasses.replace(
        state,
        unit_handler=unit_handler,
        cache=cache,
        header=state.header.with_units(convention),
    )


def redistribute(state: DatasetState, rows: DataIndex) -> DatasetState:
    required = get_all_required_pairs(state.producers, state.column_map)
    cached_columns: defaultdict = defaultdict(list)
    for uuid, name in required:
        producer = state.producers[uuid]
        if isinstance(producer, EvaluatedColumn) or (
            isinstance(producer, RawColumn) and not producer.on_disk
        ):
            cached_columns[uuid].append(name)
    comm = get_comm_world()
    assert comm is not None
    if cached_columns:
        reorder_map = None
        new_index = gather_index(rows, comm)
        original_index = gather_index(state.raw_index, comm)
        if comm.Get_rank() == 0:
            reorder_map = reindex_column(original_index, new_index)
        cache = state.cache.redistribute(reorder_map, len(rows), cached_columns, comm)
    else:
        cache = state.cache.empty()
    return dataclasses.replace(
        state, cache=cache, raw_data_handler=state.raw_data_handler.with_index(rows)
    )
