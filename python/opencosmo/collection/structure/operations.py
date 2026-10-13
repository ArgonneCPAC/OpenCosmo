"""
Structure collection operations on StructureState.

Row operations change the source's rows and defer rebuilding the linked
children (see ``StructureState.children``). Column operations on the source or
on children that commute with row selection keep the current alignment;
operations that need row-aligned children start from the resolved children.
"""

from __future__ import annotations

from dataclasses import replace
from inspect import signature
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np

from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.structure import evaluate as sevaluate
from opencosmo.collection.structure import io as sio
from opencosmo.collection.structure import links
from opencosmo.collection.structure.state import StructureState, public_keys, view
from opencosmo.column import col
from opencosmo.column.column import DerivedScalarValue
from opencosmo.column.select import do_multi_dataset_drops, do_multi_dataset_selections
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.formats import to_numpy_dict, verify_format
from opencosmo.dataset.state import DatasetState
from opencosmo.io.schema import FileEntry
from opencosmo.io.schema import make_schema as make_collection_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping
    from uuid import UUID

    import astropy.units as u
    import numpy.typing as npt

    from opencosmo.collection.structure.state import Source, StructureChild
    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region

    Leaves = Mapping[tuple[str, ...], DatasetState | LightconeState]


def __source_op(source: Source, name: str) -> Callable[..., Any]:
    """
    The operation ``name`` for a dataset or lightcone source. The dataset and
    lightcone operation modules share operation names and signatures.
    """
    if isinstance(source, DatasetState):
        return getattr(dsops, name)
    return getattr(lcops, name)


def require_lightcone(state: StructureState, method: str) -> LightconeState:
    if not isinstance(state.source, LightconeState):
        raise AttributeError(
            f"{method} is only available on lightcone structure collections."
        )
    return state.source


# Row operations


def filter(
    state: StructureState,
    *masks: ColumnMask,
    on_galaxies: bool = False,
    mode: str = "global",
) -> StructureState:
    if not masks:
        return state
    if not on_galaxies or state.dtype == "galaxy_properties":
        new_source = __source_op(state.source, "filter")(
            state.source, *masks, mode=mode
        )
        return __with_source_rows(state, new_source)
    if "galaxy_properties" not in state.linked:
        raise ValueError("Dataset galaxy_properties not found in collection.")

    galaxies = state.children["galaxy_properties"]
    assert not isinstance(galaxies, StructureState)
    filtered = __source_op(galaxies, "filter")(galaxies, *masks, mode=mode)
    matching = __source_op(galaxies, "select")(filtered, "fof_halo_tag")
    tags = __source_op(galaxies, "get_data")(matching, "astropy", True)
    new_source = __source_op(state.source, "filter")(
        state.source, col("fof_halo_tag").isin(tags)
    )
    return __with_source_rows(state, new_source)


def bound(
    state: StructureState, region: Region, select_by: str | None = None
) -> StructureState:
    new_source = __source_op(state.source, "bound")(state.source, region, select_by)
    return __with_source_rows(state, new_source)


def sort_by(state: StructureState, column: str, invert: bool = False) -> StructureState:
    new_source = __source_op(state.source, "sort_by")(state.source, column, invert)
    return __with_source_rows(state, new_source)


def take(state: StructureState, n: int, at: str, mode: str) -> StructureState:
    new_source = __source_op(state.source, "take")(state.source, n, at, mode)
    return __with_source_rows(state, new_source)


def take_range(
    state: StructureState, start: int, end: int, mode: str
) -> StructureState:
    new_source = __source_op(state.source, "take_range")(state.source, start, end, mode)
    return __with_source_rows(state, new_source)


def take_rows(state: StructureState, rows: DataIndex) -> StructureState:
    new_source = __source_op(state.source, "take_rows")(state.source, rows)
    return __with_source_rows(state, new_source)


def with_redshift_range(
    state: StructureState, z_low: float, z_high: float
) -> StructureState:
    source = require_lightcone(state, "with_redshift_range")
    return __with_source_rows(state, lcops.with_redshift_range(source, z_low, z_high))


def pixel_search(
    state: StructureState, pixels: npt.NDArray[np.int_], nside: int
) -> StructureState:
    source = require_lightcone(state, "pixel_search")
    return __with_source_rows(state, lcops.pixel_search(source, pixels, nside))


def get_pixels(state: StructureState, nside: int) -> np.ndarray:
    return lcops.get_pixels(require_lightcone(state, "get_pixels"), nside)


# Column operations


def select(
    state: StructureState,
    *select_args: str | Iterable[str],
    mode: str = "global",
    **select_kwargs: Any,
) -> StructureState:
    if not select_kwargs and not select_args:
        return state

    if select_args or not set(select_kwargs).intersection(public_keys(state)):
        if any(isinstance(s, DerivedScalarValue) for s in select_kwargs.values()):
            raise ValueError(
                "Scalar values cannot be retrieved from a StructureCollection directly. Use `collection[dataset].select(scalar_expression)`"
            )
        selected = do_multi_dataset_selections(
            leaf_datasets(state),
            select_args,
            select_kwargs,
            mode,
        )
        return rebuild_hierarchy(state, selected)

    new_source = state.source
    new_linked: dict[str, StructureChild] = {}
    for dataset, columns in select_kwargs.items():
        if isinstance(columns, dict) and (
            "columns" in columns or "derived_columns" in columns
        ):
            arg = list(columns.get("columns", []))
            kwargs = columns.get("derived_columns", {})
        else:
            arg = columns
            kwargs = {}
        if any(isinstance(c, DerivedScalarValue) for c in kwargs.values()):
            raise ValueError(
                f"Scalar values cannot be retrieved from a StructureCollection directly. Use `collection[{dataset}].select(scalar_expression)`"
            )

        if dataset == state.dtype:
            new_source = __source_op(state.source, "select")(
                state.source, arg, mode=mode, **kwargs
            )
            continue
        elif dataset not in state.linked:
            raise ValueError(f"Dataset {dataset} not found in collection.")

        child = state.linked[dataset]
        if isinstance(child, StructureState):
            if not isinstance(arg, dict):
                raise ValueError(
                    "When working with nested structure collections, the argument should be a dictionary!"
                )
            new_linked[dataset] = select(child, mode=mode, **arg)
        else:
            new_linked[dataset] = __source_op(child, "select")(
                child, arg, mode=mode, **kwargs
            )

    return replace(state, source=new_source, linked={**state.linked, **new_linked})


def drop(
    state: StructureState,
    *drop_args: str | Iterable[str],
    **columns_to_drop: Any,
) -> StructureState:
    if not drop_args and not columns_to_drop:
        return state

    if drop_args:
        dropped = do_multi_dataset_drops(
            leaf_datasets(state),
            drop_args,
        )
        state = rebuild_hierarchy(state, dropped)
        if not columns_to_drop:
            return state
        return drop(state, **columns_to_drop)

    new_source = state.source
    new_linked: dict[str, StructureChild] = {}
    for dataset, columns in columns_to_drop.items():
        if dataset == state.dtype:
            new_source = __source_op(state.source, "drop")(state.source, columns)
            continue
        elif dataset not in state.linked:
            raise ValueError(f"Dataset {dataset} not found in collection.")

        child = state.linked[dataset]
        if isinstance(child, StructureState):
            if not isinstance(columns, dict):
                raise ValueError(
                    "When working with nested structure collections, the argument should be a dictionary!"
                )
            new_linked[dataset] = drop(child, **columns)
        else:
            new_linked[dataset] = __source_op(child, "drop")(child, columns)

    return replace(state, source=new_source, linked={**state.linked, **new_linked})


def with_new_columns(
    state: StructureState,
    dataset: str,
    descriptions: str | dict[str, str] = {},
    allow_overwrite: bool = False,
    mode: str = "global",
    **new_columns: ConstructedColumn | np.ndarray,
) -> StructureState:
    if any(isinstance(c, DerivedScalarValue) for c in new_columns.values()):
        raise ValueError(
            f"Scalar values cannot be retrieved from a StructureCollection directly. Use `collection[{dataset}].select(scalar_expression)`"
        )
    path = dataset.split(".")
    if len(path) > 1:
        collection_name = path[0]
        if collection_name not in public_keys(state):
            raise ValueError(f"No collection {collection_name} found!")
        resolved = __resolved(state)
        nested = resolved.linked[collection_name]
        if not isinstance(nested, StructureState):
            raise ValueError(f"{collection_name} is not a collection!")
        new_nested = with_new_columns(
            nested,
            ".".join(path[1:]),
            descriptions=descriptions,
            allow_overwrite=allow_overwrite,
            mode=mode,
            **new_columns,
        )
        return replace(
            resolved, linked={**resolved.linked, collection_name: new_nested}
        )

    if dataset == state.dtype:
        new_source = __source_op(state.source, "with_new_columns")(
            state.source, descriptions, allow_overwrite, mode=mode, **new_columns
        )
        return replace(state, source=new_source)
    elif dataset not in state.linked:
        raise ValueError(f"Dataset {dataset} not found in this collection!")

    resolved = __resolved(state)
    child = resolved.linked[dataset]
    if not isinstance(child, DatasetState):
        raise ValueError(f"{dataset} is not a dataset!")

    new_child = dsops.with_new_columns(
        child, descriptions, allow_overwrite, mode=mode, **new_columns
    )
    in_memory = {
        name for name, column in new_columns.items() if isinstance(column, np.ndarray)
    }
    new_derived = set(new_child.columns) - set(child.columns) - in_memory
    return replace(
        resolved,
        linked={**resolved.linked, dataset: new_child},
        derived_columns=state.derived_columns.union(
            f"{dataset}.{name}" for name in new_derived
        ),
    )


def with_units(
    state: StructureState,
    convention: str | None = None,
    conversions: dict[u.Unit, u.Unit] = {},
    **dataset_conversions: dict,
) -> StructureState:
    keys = public_keys(state)
    if conversions:
        dataset_conversions = dict(dataset_conversions)
        for name in keys:
            ds_conversions = dict(dataset_conversions.get(name, {}))
            ds_conversions["conversions"] = conversions | ds_conversions.get(
                "conversions", {}
            )
            dataset_conversions[name] = ds_conversions

    unknown = set(dataset_conversions).difference(keys)
    if unknown:
        raise ValueError(f"Unknown datasets in conversions: {unknown}")

    new_source = state.source
    if state.dtype in dataset_conversions or (
        not dataset_conversions and convention is not None
    ):
        new_source = __with_source_units(
            state.source, convention, dataset_conversions.get(state.dtype, {})
        )

    new_linked = {
        name: __with_child_units(child, convention, dataset_conversions.get(name, {}))
        for name, child in state.linked.items()
    }
    return replace(state, source=new_source, linked=new_linked)


def with_datasets(
    state: StructureState, datasets: str | Iterable[str]
) -> StructureState:
    requested = {datasets} if isinstance(datasets, str) else set(datasets)
    known = set(public_keys(state))
    if not requested.issubset(known):
        raise ValueError(f"Unknown datasets {requested - known}")

    return replace(
        state,
        linked={
            name: child for name, child in state.linked.items() if name in requested
        },
        hide_source=state.dtype not in requested,
    )


# Iteration and evaluation


def iter_objects(
    state: StructureState, data_types: Iterable[str] | None = None
) -> Generator[dict[str, StructureChild | dict[str, Any]]]:
    """
    Yield one structure at a time: its source row as a dict and the slice of
    each requested linked child that belongs to it. After a complete
    iteration, derived columns collected along the way are written into the
    linked datasets' caches so they are not recomputed.
    """
    data_types = list(state.linked if data_types is None else data_types)
    if not all(name in state.linked for name in data_types):
        raise ValueError("Some data types are not linked in the collection.")
    if len(state) == 0:
        warn("Tried to iterate over a collection with no structures in it!")
        return

    children = state.children
    link_names = set(links.link_names(state.match_sets))
    slots = {
        name: links.slot_values(state.match_sets, state.source, name)
        for name in data_types
        if name in link_names
    }
    starts = dict.fromkeys(slots, 0)
    collected: dict[str, dict[str, list[np.ndarray]]] = {}
    for column in sorted(state.derived_columns):
        dataset, name = column.split(".", maxsplit=1)
        collected.setdefault(dataset, {})[name] = []

    try:
        rows = __source_op(state.source, "rows")(state.source)
        for i, row in enumerate(rows):
            output: dict[str, StructureChild | dict[str, Any]] = {}
            for name, (values, is_chunked) in slots.items():
                length = int(values[i]) if is_chunked else int(values[i] >= 0)
                if length == 0:
                    continue
                start = starts[name]
                output[name] = __take_range(children[name], start, start + length)
                starts[name] += length

            if not state.hide_source:
                output[state.dtype] = dict(row)
            if not output:
                continue

            for dataset, columns in collected.items():
                piece = output.get(dataset)
                if piece is None:
                    continue
                assert isinstance(piece, DatasetState)
                names = set(piece.columns).intersection(columns)
                if not names:
                    continue
                data = dsops.get_data(
                    dsops.select(piece, names),
                    "astropy",
                    unpack=False,
                    wrap_single=True,
                )
                for name in data.colnames:
                    columns[name].append(data[name])

            yield output

        __cache_collected(children, collected)
    except GeneratorExit:
        return


def evaluate(
    state: StructureState,
    func: Callable,
    dataset: str | None = None,
    format: str = "astropy",
    insert: bool = True,
    allow_overwrite: bool = False,
    **evaluate_kwargs: Any,
) -> StructureState | dict[str, Any] | None:
    # Three cases:
    # 1. Evaluate on several datasets and insert into the source.
    # 2. Evaluate on a single dataset and insert into that dataset (lazy).
    # 3. Evaluate on several datasets and insert into one of them.
    # Cases 1 and 3 are eager, as is everything when insert=False.
    if dataset is not None and dataset == state.dtype:
        return evaluate_on_dataset(
            state,
            func,
            dataset=dataset,
            format=format,
            insert=insert,
            allow_overwrite=allow_overwrite,
            **evaluate_kwargs,
        )

    verify_format(format)

    if dataset is not None and dataset.startswith("galaxies"):
        dataset_path = dataset.split(".")
        sub_dataset = dataset_path[1] if len(dataset_path) == 2 else None
        resolved = __resolved(state)
        nested = view(resolved, dataset_path[0])
        assert isinstance(nested, StructureState)
        result = evaluate(
            nested,
            func,
            sub_dataset,
            format,
            insert,
            allow_overwrite=allow_overwrite,
            **evaluate_kwargs,
        )
        if not insert:
            return result
        assert isinstance(result, StructureState)
        return replace(resolved, linked={**resolved.linked, dataset_path[0]: result})

    keys = public_keys(state)
    required_datasets = set(signature(func).parameters).intersection(keys)
    if not required_datasets and dataset is None:
        raise ValueError(
            "If your function does not take dataset names as arguments, you must specify which dataset you want to evaluate on!"
        )

    if dataset is not None and not required_datasets:
        # Case 2
        if dataset not in keys:
            raise ValueError(f"Unknown dataset {dataset}")
        resolved = __resolved(state)
        child = view(resolved, dataset)
        assert not isinstance(child, StructureState)
        result = __source_op(child, "evaluate")(
            child,
            func,
            insert=insert,
            format=format,
            allow_overwrite=allow_overwrite,
            strategy="chunked",
            **evaluate_kwargs,
        )
        if not insert:
            return result
        assert isinstance(child, DatasetState) and isinstance(result, DatasetState)
        new_derived = set(result.columns).difference(child.columns)
        return replace(
            resolved,
            linked={**resolved.linked, dataset: result},
            derived_columns=state.derived_columns.union(
                f"{dataset}.{name}" for name in new_derived
            ),
        )

    # Cases 1 and 3
    output = sevaluate.visit_structure_eagerly(
        func,
        state,
        dataset=dataset,
        format=format,
        evaluate_kwargs=evaluate_kwargs,
        insert=insert,
    )
    if not insert or output is None:
        return output
    return with_new_columns(
        state,
        dataset if dataset is not None else state.dtype,
        allow_overwrite=allow_overwrite,
        **to_numpy_dict(output),  # type: ignore[arg-type]
    )


def evaluate_on_dataset(
    state: StructureState,
    func: Callable,
    dataset: str | None = None,
    vectorize: bool = False,
    format: str = "astropy",
    insert: bool = True,
    batch_size: int = -1,
    allow_overwrite: bool = False,
    **evaluate_kwargs: Any,
) -> StructureState | dict[str, Any] | None:
    if dataset is None or dataset == state.dtype:
        result = __source_op(state.source, "evaluate")(
            state.source,
            func,
            vectorize=vectorize,
            insert=insert,
            format=format,
            batch_size=batch_size,
            allow_overwrite=allow_overwrite,
            **evaluate_kwargs,
        )
        if not insert:
            return result
        return replace(state, source=result)

    path = dataset.split(".")
    if path[0] not in state.linked:
        raise ValueError(f"Unknown dataset {dataset}")
    child = state.linked[path[0]]

    if not isinstance(child, StructureState):
        if len(path) > 1:
            raise ValueError(
                f"Recieved {dataset} as the dataset argument but {path[0]} is a dataset, not a collection!"
            )
        result = __source_op(child, "evaluate")(
            child,
            func,
            vectorize=vectorize,
            insert=insert,
            format=format,
            batch_size=batch_size,
            allow_overwrite=allow_overwrite,
            **evaluate_kwargs,
        )
        if not insert:
            return result
        new_derived = set()
        if isinstance(child, DatasetState):
            new_derived = set(result.columns).difference(child.columns)
        return replace(
            state,
            linked={**state.linked, dataset: result},
            derived_columns=state.derived_columns.union(
                f"{dataset}.{name}" for name in new_derived
            ),
        )

    if len(path) == 1:
        nested_result = evaluate(
            child, func, None, format, insert, allow_overwrite=allow_overwrite
        )
    else:
        nested_result = evaluate_on_dataset(
            child,
            func,
            ".".join(path[1:]),
            vectorize,
            format,
            insert,
            batch_size=batch_size,
            allow_overwrite=allow_overwrite,
        )
    if not insert:
        return nested_result
    assert isinstance(nested_result, StructureState)
    return replace(state, linked={**state.linked, path[0]: nested_result})


# Opening and writing


def open_state(
    targets: list[DatasetTarget],
    ignore_empty: bool = True,
    index_kind: str = "none",
    is_empty_ref: bool = False,
) -> StructureState:
    return sio.build_structure_collection(
        targets, ignore_empty, index_kind=index_kind, is_empty_ref=is_empty_ref
    )


def make_schema(state: StructureState, path: str) -> Schema:
    """
    Schema for writing a structure collection. Linked children are restored to
    the source's file order, and the source carries the link columns.
    """
    no_stack = isinstance(state.source, LightconeState)
    children: dict[str, Schema] = {}

    # Every rank builds the source schema, even when empty, to stay in lockstep
    # with the collectives inside it.
    source_schema = child_schema(state.source, "/".join([path, state.dtype]), no_stack)
    if len(state.source) > 0:
        source_schema = links.link_schema(state.match_sets, state.source, source_schema)
        children[state.dtype] = sio.rebuild_data_linked(source_schema)

    resorted = links.resort(state.match_sets, state.source, state.children)
    for name in sorted(resorted):
        child = resorted[name]
        write_name = "galaxy_properties" if name == "galaxies" else name
        schema = child_schema(child, "/".join([path, write_name]), no_stack)
        if not isinstance(child, StructureState):
            children[write_name] = schema
            continue
        for grandchild_name, grandchild in schema.children.items():
            if "properties" in grandchild_name:
                children[grandchild_name] = grandchild
            else:
                children[f"{write_name}_{grandchild_name}"] = grandchild

    return make_collection_schema(
        path.split("/")[-1], FileEntry.STRUCTURE_COLLECTION, children=children
    )


def leaf_datasets(
    state: StructureState, path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], DatasetState | LightconeState]:
    """Every dataset or lightcone in the hierarchy, keyed by path."""
    leaves: dict[tuple[str, ...], DatasetState | LightconeState] = {}
    if not state.hide_source:
        leaves[path + (state.dtype,)] = state.source
    for name, child in state.children.items():
        if isinstance(child, StructureState):
            leaves.update(leaf_datasets(child, path + (name,)))
        else:
            leaves[path + (name,)] = child
    return leaves


def rebuild_hierarchy(
    state: StructureState, leaves: Leaves, path: tuple[str, ...] = ()
) -> StructureState:
    """Replace the leaves of a resolved hierarchy, keeping its shape."""
    new_source = leaves.get(path + (state.dtype,), state.source)
    new_linked: dict[str, StructureChild] = {}
    for name, child in state.children.items():
        child_path = path + (name,)
        if isinstance(child, StructureState):
            new_linked[name] = rebuild_hierarchy(child, leaves, child_path)
        else:
            new_linked[name] = leaves[child_path]
    return replace(state, source=new_source, linked=new_linked, linked_source=None)


def __with_child_units(
    child: StructureChild, convention: str | None, conversions: dict
) -> StructureChild:
    if isinstance(child, StructureState):
        return with_units(child, convention, **conversions)
    return __with_source_units(child, convention, conversions)


def __with_source_units(
    source: Source, convention: str | None, conversions: dict
) -> Source:
    column_conversions = dict(conversions)
    unit_conversions = column_conversions.pop("conversions", {})
    return __source_op(source, "with_units")(
        source, convention, unit_conversions, **column_conversions
    )


def __with_source_rows(state: StructureState, new_source: Source) -> StructureState:
    linked_source = state.linked_source
    if linked_source is None:
        linked_source = state.source
    return replace(state, source=new_source, linked_source=linked_source)


def __resolved(state: StructureState) -> StructureState:
    return replace(state, linked=state.children, linked_source=None)


def __take_range(child: StructureChild, start: int, end: int) -> StructureChild:
    if isinstance(child, StructureState):
        return take_range(child, start, end, "local")
    return __source_op(child, "take_range")(child, start, end, "local")


def __cache_collected(
    children: Mapping[str, StructureChild],
    collected: Mapping[str, Mapping[str, list[np.ndarray]]],
) -> None:
    """Store per-structure derived values in the linked datasets' caches."""
    for dataset, columns in collected.items():
        child = children[dataset]
        assert isinstance(child, DatasetState)
        by_uuid: dict[UUID, dict[str, np.ndarray]] = {}
        for name, pieces in columns.items():
            if not pieces:
                continue
            values = np.concatenate(pieces)
            if len(values) > len(child):
                values = values.reshape((len(child), -1))
            by_uuid.setdefault(child.column_map[name], {})[name] = values
        child.cache.add_data(by_uuid, {})


# Lightcones are written in stacks of at least this many rows, as Lightcone does.
__MIN_LIGHTCONE_DATASET_SIZE = 100_000


def child_schema(child: StructureChild, path: str, no_stack: bool) -> Schema:
    if isinstance(child, DatasetState):
        return dsops.make_schema(child, path)
    if isinstance(child, LightconeState):
        return lcops.make_schema(child, path, __MIN_LIGHTCONE_DATASET_SIZE, no_stack)
    return make_schema(child, path)
