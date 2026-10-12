from __future__ import annotations

import dataclasses
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import healpy as hp
import numpy as np
from astropy.table import vstack  # type: ignore

from opencosmo.collection.lightcone import io as lcio
from opencosmo.collection.lightcone import state as lcst
from opencosmo.collection.lightcone import utils as lcutils
from opencosmo.collection.lightcone.healpix_map import HealpixMap
from opencosmo.collection.lightcone.instantiate import evaluate_scope
from opencosmo.collection.lightcone.scope import partition_columns
from opencosmo.collection.lightcone.stack import stack_lightcone_datasets_in_schema
from opencosmo.column import col
from opencosmo.column.column import (
    Column,
    DerivedScalarValue,
    EvaluatedColumn,
    mask_contains_scalar,
    resolve_mask_scalars,
)
from opencosmo.column.reducer import default_reducer
from opencosmo.column.select import build_multi_dataset_selections
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.evaluate import build_evaluated_column
from opencosmo.dataset.formats import (
    average_chunks,
    concat_chunks,
    convert_data,
    product_chunks,
    sum_chunks,
    verify_format,
)
from opencosmo.dataset.take import (
    get_end_take_index,
    get_random_take_index,
    get_range_take_index,
    get_rows_take_index,
)
from opencosmo.index import empty, get_range, into_array, rebuild_by_ranges
from opencosmo.io import iopen, specs
from opencosmo.io.index_spec import index_spec_for
from opencosmo.io.schema import FileEntry
from opencosmo.io.schema import make_schema as make_collection_schema
from opencosmo.mpi import get_all_keys, get_comm_world
from opencosmo.plugins.contexts import (
    HookPoint,
    LightconeInstantiateCtx,
    LightconeOpenCtx,
    PostSortCtx,
)
from opencosmo.plugins.hook import fold
from opencosmo.spatial.index import project_on_index

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    import astropy.units as u
    import numpy.typing as npt

    from opencosmo.collection.lightcone.scope import LightconeScope
    from opencosmo.collection.lightcone.state import LeafKey, LightconeState
    from opencosmo.column.column import (
        ColumnMask,
        CompoundColumnMask,
        ConstructedColumn,
    )
    from opencosmo.dataset.state import DatasetState
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region

    Leaves = tuple[tuple[LeafKey, DatasetState], ...]


def open_state(
    targets: list[DatasetTarget],
    index_kind: str = "none",
    is_empty_ref: bool = False,
    **open_kwargs: Any,
) -> LightconeState:
    """
    Open lightcone datasets (and any companion healpix maps) from ``targets``.
    Steps with several datasets get ``(step, name)`` keys.
    """
    groups: dict[int, dict[str, DatasetState]] = defaultdict(dict)
    maps: list[DatasetTarget] = []
    for i, target in enumerate(targets):
        if target.header.file.data_type == "healpix_map":
            maps.append(target)
            continue

        step = target.header.file.step
        name = target.name.removeprefix(f"{step}_")
        dataset = iopen.open_dataset(
            target,
            index_spec_for(index_kind, is_empty_ref, is_source=True),
            open_kwargs=open_kwargs,
        )
        groups[step if step is not None else i][name] = dataset._state

    leaves = [
        ((step,), leaf) if len(group) == 1 else ((step, name), leaf)
        for step, group in groups.items()
        for name, leaf in group.items()
    ]

    healpix_maps = None
    if maps:
        # A map opened alongside a catalog is replicated, not partitioned: a
        # cutout around an object near a partition edge needs map pixels that
        # this rank's catalog rows do not cover.
        healpix_maps = specs.HealpixMapSpec().build_from_targets(
            maps,
            index_kind=index_kind,
            is_empty_ref=is_empty_ref,
            open_kwargs=open_kwargs,
            index=index_spec_for(
                index_kind, is_empty_ref, is_source=True, is_replicated=True
            ),
        )
    assert healpix_maps is None or isinstance(healpix_maps, HealpixMap)

    return from_dataset_states(leaves, maps=healpix_maps, **open_kwargs)


def from_dataset_states(
    leaves: Iterable[tuple[LeafKey, DatasetState]],
    maps: HealpixMap | None = None,
    z_range: tuple[float, float] | None = None,
    scope: LightconeScope | None = None,
    **open_kwargs: Any,
) -> LightconeState:
    """Build a lightcone from ``leaves`` and run the lightcone open hooks."""
    state = lcst.from_leaves(leaves, maps, z_range, scope=scope)
    return fold(HookPoint.LightconeOpen, LightconeOpenCtx(state, open_kwargs)).state


def with_units(
    state: LightconeState,
    convention: str | None,
    conversions: dict[u.Unit, u.Unit],
    **columns: u.Unit,
) -> LightconeState:
    return __map_leaves(
        state, lambda leaf: dsops.with_units(leaf, convention, conversions, **columns)
    )


def filter(
    state: LightconeState,
    *masks: ColumnMask | CompoundColumnMask,
    mode: str = "global",
) -> LightconeState:
    if not masks:
        return state
    evaluator = __scalar_evaluator(state, mode)
    masks = tuple(
        resolve_mask_scalars(mask, evaluator) if mask_contains_scalar(mask) else mask
        for mask in masks
    )
    return __map_leaves(state, lambda leaf: dsops.filter(leaf, *masks, mode=mode))


def get_data(
    state: LightconeState,
    format: str = "astropy",
    unpack: bool = True,
    wrap_single: bool = False,
    ignore_sort: bool = False,
):
    verify_format(format)
    instantiated = fold(
        HookPoint.LightconeInstantiate, LightconeInstantiateCtx(state)
    ).state

    data = [
        dsops.get_data(leaf, "astropy", unpack=False, wrap_single=True)
        for _, leaf in instantiated.leaves
    ]
    data_with_length = [d for d in data if len(d) > 0]
    if len(data_with_length) == 0:
        vstacked = data[0]
    else:
        vstacked = vstack(data_with_length, join_type="exact")

    vstacked, scope_scalars = evaluate_scope(state.scope, vstacked, unpack=unpack)
    if scope_scalars is not None:
        return convert_data(scope_scalars, format, wrap_single=wrap_single)

    if state.sort_key is not None and not ignore_sort:
        order = vstacked.argsort(state.sort_key[0], reverse=state.sort_key[1])
        vstacked = vstacked[order]
        vstacked = fold(
            HookPoint.PostSort, PostSortCtx(state, vstacked, np.argsort(order))
        ).data

    vstacked.remove_columns(state.hidden.intersection(vstacked.colnames))
    if len(vstacked) == 1 and unpack:
        output_data = {
            key: value[0] if len(value) == 1 else value
            for key, value in vstacked.items()
        }
        return convert_data(output_data, format, wrap_single=wrap_single)

    if format != "astropy":
        return convert_data(dict(vstacked), format, wrap_single=wrap_single)
    elif len(vstacked.columns) == 1 and not wrap_single:
        return next(iter(dict(vstacked).values()))

    return vstacked


def rows(state: LightconeState) -> Generator[dict[str, float | u.Quantity]]:
    yield from chain.from_iterable(dsops.rows(leaf) for _, leaf in state.leaves)


def bound(
    state: LightconeState, region: Region, select_by: str | None
) -> LightconeState:
    new_maps = None if state.maps is None else state.maps.bound(region)
    return __map_leaves(
        state,
        lambda leaf: dsops.bound(leaf, region, select_by),
        maps=new_maps or state.maps,
    )


def sort_by(state: LightconeState, column: str | None, invert: bool) -> LightconeState:
    if column is None:
        return dataclasses.replace(state, sort_key=None)
    if column not in state.columns:
        raise ValueError(f"Column {column} does not exist in this dataset!")
    return dataclasses.replace(state, sort_key=(column, invert))


def with_redshift_range(
    state: LightconeState, z_low: float, z_high: float
) -> LightconeState:
    if z_high < z_low:
        z_high, z_low = z_low, z_high

    if z_high < state.z_range[0] or z_low > state.z_range[1]:
        return take_rows_unsorted(state, empty())
    elif z_low == z_high:
        raise ValueError("Low and high values of the redshift range are the same!")

    in_range = col("redshift") > z_low, col("redshift") < z_high
    leaves: list[tuple[LeafKey, DatasetState]] = []
    for step, group in __group_by_step(state.leaves).items():
        ranges = [lcst.header_z_range(leaf.header) for _, leaf in group]
        if z_high < min(r[0] for r in ranges) or z_low > max(r[1] for r in ranges):
            continue
        filtered = tuple((key, dsops.filter(leaf, *in_range)) for key, leaf in group)
        leaves.extend(((step, *key), leaf) for key, leaf in __prune_empty(filtered))

    return __with_leaves(state, tuple(leaves), z_range=(z_low, z_high))


def pixel_search(
    state: LightconeState, pixels: npt.NDArray[np.int_], nside: int
) -> LightconeState:
    level = np.log2(nside)
    if not level.is_integer() or level < 0:
        raise ValueError("nside must be a positive power of two!")
    level = int(level)
    pixels = np.unique(np.atleast_1d(pixels))
    if not np.isdtype(pixels.dtype, "integral") or len(pixels) == 0:
        raise ValueError("Pixels must be a 1d array of positive integers")
    if pixels[0] < 0 or pixels[-1] >= hp.nside2npix(nside):
        raise ValueError("Pixels must be a 1d array of positive integers")

    leaves = []
    for key, leaf in state.leaves:
        if leaf.spatial_index is None:
            raise ValueError("Lightcone does not have a spatial index!")
        rows = project_on_index(leaf.spatial_index, level, leaf.raw_index, pixels)
        leaves.append((key, dsops.take_rows(leaf, rows)))
    return __with_leaves(state, tuple(leaves))


def select(
    state: LightconeState,
    *columns: str | Iterable[str],
    mode: str = "global",
    **derived_columns: ConstructedColumn,
) -> LightconeState:
    if state.maps is not None:
        selection_args, selection_kwargs = build_multi_dataset_selections(
            {"map": set(state.maps.columns), "lightcone": set(state.columns)},
            {"map": len(state.maps), "lightcone": len(state)},
            columns,
            derived_columns,
        )
        lightcone_columns = selection_args["lightcone"]
        lightcone_derived_columns = selection_kwargs["lightcone"]
        new_maps = state.maps.select(*selection_args["map"], **selection_kwargs["map"])
    else:
        lightcone_columns = columns
        lightcone_derived_columns = derived_columns
        new_maps = None

    all_columns: set[str] = set()
    for col_group in lightcone_columns:
        all_columns.update({col_group} if isinstance(col_group, str) else col_group)

    has_scalars = any(
        isinstance(c, DerivedScalarValue) for c in lightcone_derived_columns.values()
    )
    has_non_scalars = any(
        not isinstance(c, DerivedScalarValue)
        for c in lightcone_derived_columns.values()
    )
    if has_scalars and (all_columns or has_non_scalars):
        raise ValueError(
            "Scalar selections cannot be mixed with column selections. "
            "Call select() with only scalar kwargs, or only column selections."
        )

    reducer = default_reducer(mode)
    lightcone_derived_columns = {
        k: v.with_reducer(reducer) if isinstance(v, (Column, DerivedScalarValue)) else v
        for k, v in lightcone_derived_columns.items()
    }

    raw_child_columns = set(state.leaves[0][1].columns)
    plan = state.scope.plan_select(
        all_columns, lightcone_derived_columns, raw_child_columns
    )

    # Some columns must be retained even when the user does not ask for them,
    # otherwise the lightcone cannot be written. They are kept but hidden.
    required = lcutils.get_required_columns(
        raw_child_columns, state.header.parameters["dtype"]
    )
    if state.sort_key is not None:
        required.add(state.sort_key[0])

    required_extras = required.difference(all_columns)
    leaf_columns = plan.child_positional | plan.child_additional | required_extras
    selected = __map_leaves(
        state,
        lambda leaf: dsops.select(leaf, leaf_columns, mode=mode, **plan.child_scoped),
        maps=new_maps or state.maps,
    )
    return dataclasses.replace(
        selected,
        hidden=state.hidden | required_extras,
        scope=plan.new_scope,
    )


def drop(state: LightconeState, *columns: str | Iterable[str]) -> LightconeState:
    dropped_columns: set[str] = set()
    for col_group in columns:
        dropped_columns.update({col_group} if isinstance(col_group, str) else col_group)

    current_columns = set(state.columns)
    if missing := dropped_columns.difference(current_columns):
        raise ValueError(
            f"Tried to drop columns that are not in this dataset: {missing}"
        )
    return select(state, current_columns - dropped_columns)


def with_new_columns(
    state: LightconeState,
    descriptions: str | dict[str, str] = {},
    allow_overwrite: bool = False,
    mode: str = "global",
    **columns: ConstructedColumn | np.ndarray | u.Quantity,
) -> LightconeState:
    if any(isinstance(col, DerivedScalarValue) for col in columns.values()):
        raise ValueError(
            "Scalar values cannot be added to an existing dataset, but can be retrieved with Dataset.select()"
        )
    reducer = default_reducer(mode)
    columns = {
        k: v.with_reducer(reducer) if isinstance(v, Column) else v
        for k, v in columns.items()
    }
    derived: dict[str, ConstructedColumn] = {}
    raw: dict[str, np.ndarray] = {}
    for name, column in columns.items():
        if isinstance(column, (Column, EvaluatedColumn)):
            derived[name] = column
        elif not isinstance(column, np.ndarray):
            raise ValueError(f"Invalid column type: {type(columns)}")
        elif len(column) != len(state):
            raise ValueError(
                f"New column {name} has length {len(column)} but this dataset "
                f"has length {len(state)}"
            )
        else:
            raw[name] = column

    if (sort_index := get_sort_index(state)) is not None:
        inverse = np.argsort(sort_index)
        raw = {name: raw_data[inverse] for name, raw_data in raw.items()}

    raw_child_columns = set(state.leaves[0][1].columns) | set(raw.keys())
    _, child_scoped = partition_columns(derived, raw_child_columns, state.scope)
    new_scope = state.scope.add(derived, raw_child_columns)

    split_points = np.cumsum([len(leaf) for _, leaf in state.leaves])[:-1]
    raw_split = {name: np.split(arr, split_points) for name, arr in raw.items()}
    leaves = tuple(
        (
            key,
            dsops.with_new_columns(
                leaf,
                descriptions,
                allow_overwrite,
                mode=mode,
                **{name: arrs[i] for name, arrs in raw_split.items()},
                **child_scoped,
            ),
        )
        for i, (key, leaf) in enumerate(state.leaves)
    )
    return dataclasses.replace(__with_leaves(state, leaves), scope=new_scope)


def evaluate(
    state: LightconeState,
    func: Callable,
    vectorize: bool = False,
    insert: bool = True,
    format: str = "astropy",
    batch_size: int = -1,
    allow_overwrite: bool = False,
    combine_mode: Literal["concatenate", "sum", "prod", "avg"] = "concatenate",
    **evaluate_kwargs: Any,
) -> LightconeState | dict[str, Any] | None:
    if combine_mode not in ("concatenate", "sum", "prod", "avg"):
        raise ValueError(f"Unknown combine mode {combine_mode}")

    steps = set(lcst.public_keys(state))
    mapped_kwargs = {
        name: evaluate_kwargs.pop(name)
        for name, value in list(evaluate_kwargs.items())
        if isinstance(value, dict) and set(value.keys()) == steps
    }

    def leaf_kwargs(key: LeafKey) -> dict[str, Any]:
        return {name: values[key[0]] for name, values in mapped_kwargs.items()}

    if insert:
        first_key, first_leaf = state.leaves[0]
        evaluated_column = build_evaluated_column(
            first_leaf,
            func,
            vectorize,
            insert,
            format,
            batch_size,
            evaluate_kwargs | leaf_kwargs(first_key),
        )
        leaves = tuple(
            (
                key,
                dsops.with_new_columns(
                    leaf,
                    {},
                    allow_overwrite,
                    **{func.__name__: evaluated_column.with_kwargs(**leaf_kwargs(key))},
                ),
            )
            for key, leaf in state.leaves
        )
        # Nested steps never pruned their own subtypes here; only steps are pruned.
        return __with_leaves(state, __prune_empty(leaves, depth=1))

    results = [
        dsops.evaluate(
            leaf,
            func,
            vectorize,
            insert,
            format,
            batch_size,
            allow_overwrite,
            **evaluate_kwargs,
            **leaf_kwargs(key),
        )
        for key, leaf in state.leaves
    ]
    if results[0] is None:
        return None

    chunks_by_key = {key: [result[key] for result in results] for key in results[0]}
    if combine_mode != "concatenate":
        for key, chunks in chunks_by_key.items():
            lengths = [len(chunk) for chunk in chunks]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Cannot combine evaluated output '{key}' with "
                    f"combine_mode='{combine_mode}': all underlying dataset "
                    f"results must have the same length; got {lengths}"
                )

    output = {}
    for key, chunks in chunks_by_key.items():
        match combine_mode:
            case "concatenate":
                output[key] = concat_chunks(chunks, format)
            case "sum":
                output[key] = sum_chunks(chunks, format)
            case "prod":
                output[key] = product_chunks(chunks, format)
            case "avg":
                weights = [len(leaf) for _, leaf in state.leaves]
                output[key] = average_chunks(chunks, weights, format)
    return output


def take(
    state: LightconeState, n: int, at: str, mode: Literal["local", "global"]
) -> LightconeState:
    if at == "random":
        return take_rows_unsorted(state, get_random_take_index(n, len(state), mode))
    elif at == "start":
        index = get_range_take_index(state, state.sort_key, 0, n, mode)
    elif at == "end":
        index = get_end_take_index(n, state, state.sort_key, mode)
    else:
        raise ValueError(f'"at" should be one of ("start", "end", "random", got {at}')
    return take_rows_unsorted(state, __sorted_to_physical(state, index))


def take_range(
    state: LightconeState, start: int, end: int, mode: Literal["local", "global"]
) -> LightconeState:
    if start < 0:
        raise ValueError("Tried to take negative rows!")
    index = get_range_take_index(state, state.sort_key, start, end - start, mode)
    return take_rows_unsorted(state, __sorted_to_physical(state, index))


def take_rows(state: LightconeState, rows: DataIndex) -> LightconeState:
    index_range = get_range(rows)
    if index_range[0] < 0 or index_range[1] > len(state):
        raise ValueError("Rows must be between 0 and the length of this dataset - 1")
    rows = get_rows_take_index(state, rows, state.sort_key)
    return take_rows_unsorted(state, rows)


def get_sort_index(state: LightconeState) -> np.ndarray | None:
    """Physical row positions in sorted order, or None if the lightcone is unsorted."""
    if state.sort_key is None:
        return None
    column, invert = state.sort_key
    data = np.concatenate(
        [
            dsops.get_data(dsops.select(leaf, column), "numpy")
            for _, leaf in state.leaves
        ]
    )
    if invert:
        data = -data
    return np.argsort(data)


def take_rows_unsorted(state: LightconeState, rows: DataIndex) -> LightconeState:
    """
    Take rows by physical (stacked, unsorted) position. ``rows`` is assumed to be
    sorted and in range.
    """
    sizes = np.fromiter((len(leaf) for _, leaf in state.leaves), dtype=np.int64)
    starts = np.zeros_like(sizes)
    starts[1:] = np.cumsum(sizes)[:-1]
    projected = rebuild_by_ranges(rows, (starts, sizes))
    leaves = tuple(
        (key, dsops.take_rows(leaf, index))
        for (key, leaf), index in zip(state.leaves, projected)
    )
    return __with_leaves(state, __prune_empty(leaves, keep_all_empty=False))


def make_schema(
    state: LightconeState, path: str, min_size: int, no_stack: bool
) -> Schema:
    output_datasets = lcio.combine_adjacent_datasets(
        lcio.order_by_redshift_range(state.leaves),
        min_dataset_size=min_size,
        no_stack=no_stack,
    )
    children = {}

    # Use get_all_keys to iterate over the union of steps across all ranks
    # This ensures every rank calls stack_lightcone_datasets_in_schema the same
    # number of times in the same order, preventing MPI deadlock
    all_steps = get_all_keys(output_datasets, get_comm_world())

    for step in all_steps:
        datasets = output_datasets.get(step, {})
        if len(datasets) == 0:
            # This rank has no data for this step, but every rank must still
            # enter the collectives inside stack_lightcone_datasets_in_schema
            # (get_all_keys / get_stacked_lightcone_order / sync_headers) in
            # lockstep, or cross-rank collectives mispair across steps.
            stack_lightcone_datasets_in_schema(
                datasets, "/".join([path, str(step)]), None, no_stack
            )
            continue

        header_zrange = lcst.redshift_range(chain.from_iterable(datasets.values()))
        zrange = (
            max(header_zrange[0], state.z_range[0]),
            min(header_zrange[1], state.z_range[1]),
        )

        child_schemas = stack_lightcone_datasets_in_schema(
            datasets, "/".join([path, str(step)]), zrange, no_stack
        )
        children.update(
            {f"{step}_{name}": schema for name, schema in child_schemas.items()}
        )

    comm = get_comm_world()
    rank = 0 if comm is None else comm.Get_rank()
    if state.maps is not None:
        healpix_schema = state.maps.make_schema("/".join([path, "healpix_maps"]))
        if rank == 0:
            children["healpix_maps"] = healpix_schema
    name = path.split("/")[-1]
    return make_collection_schema(name, FileEntry.LIGHTCONE, children=children)


def __scalar_evaluator(
    state: LightconeState, mode: str
) -> Callable[[DerivedScalarValue], Any]:
    """
    Build a function that evaluates a single DerivedScalarValue against
    ``state``. Used for eager resolution in filter masks.
    """
    reducer = default_reducer(mode)

    def evaluate(scalar: DerivedScalarValue) -> Any:
        scalar = scalar.with_reducer(reducer) if scalar.reducer is None else scalar
        required = scalar._traverse_names()
        if not required:
            return scalar.evaluate({})
        # `required` is the set of raw child names this scalar reads, so
        # select(state, *required) routes through the positional (child)
        # path — partition_columns treats no derived_columns as
        # all-child-scoped — and never re-enters the scope evaluator.
        # If a future caller passes a scope-name in `required`, that
        # would recurse through evaluate_scope; bypass via take_rows or
        # a private raw read if that becomes possible.
        table = get_data(
            select(state, *required), "astropy", unpack=False, wrap_single=True
        )
        return scalar.evaluate({name: table[name] for name in required})

    return evaluate


def __map_leaves(
    state: LightconeState,
    fn: Callable[[DatasetState], DatasetState],
    maps: HealpixMap | None = None,
) -> LightconeState:
    leaves = tuple((key, fn(leaf)) for key, leaf in state.leaves)
    return __with_leaves(state, __prune_empty(leaves), maps=maps)


def __with_leaves(
    state: LightconeState,
    leaves: Leaves,
    maps: HealpixMap | None = None,
    z_range: tuple[float, float] | None = None,
) -> LightconeState:
    """Rebuild ``state`` around new leaves; ``None`` arguments keep current values."""
    return lcst.from_leaves(
        leaves,
        maps if maps is not None else state.maps,
        z_range if z_range is not None else state.z_range,
        state.hidden,
        state.sort_key,
        state.scope,
    )


def __group_by_step(leaves: Leaves) -> dict[int | str, Leaves]:
    groups: dict[int | str, list[tuple[LeafKey, DatasetState]]] = defaultdict(list)
    for key, leaf in leaves:
        groups[key[0]].append((key[1:], leaf))
    return {step: tuple(group) for step, group in groups.items()}


def __prune_empty(
    leaves: Leaves, keep_all_empty: bool = True, depth: int | None = None
) -> Leaves:
    """
    Drop empty groups at each key level, down to ``depth`` levels (all levels if
    None). If every group at a level is empty, keep all of them
    (``keep_all_empty``) or only the first.
    """
    if not leaves[0][0] or depth == 0:
        return leaves
    child_depth = None if depth is None else depth - 1
    pruned = {
        step: __prune_empty(group, keep_all_empty, child_depth)
        for step, group in __group_by_step(leaves).items()
    }
    kept = {
        step: group
        for step, group in pruned.items()
        if any(len(leaf) for _, leaf in group)
    }
    if not kept:
        kept = pruned if keep_all_empty else dict([next(iter(pruned.items()))])
    return tuple(
        ((step, *key), leaf) for step, group in kept.items() for key, leaf in group
    )


def __sorted_to_physical(state: LightconeState, index: DataIndex) -> DataIndex:
    sort_index = get_sort_index(state)
    if sort_index is None:
        return index
    return np.sort(sort_index[into_array(index)])
