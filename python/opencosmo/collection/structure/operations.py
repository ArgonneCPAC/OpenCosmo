"""
Structure collection operations on StructureState.

Row operations change the source's rows and defer rebuilding the linked
children (see ``StructureState.children``). Column operations on the source or
on children that commute with row selection keep the current alignment;
operations that need row-aligned children start from the resolved children.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.structure.state import StructureState, public_keys
from opencosmo.column import col
from opencosmo.column.column import DerivedScalarValue
from opencosmo.column.select import do_multi_dataset_drops, do_multi_dataset_selections
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.state import DatasetState

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import astropy.units as u
    import numpy.typing as npt

    from opencosmo.collection.structure.state import Source, StructureChild
    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.index import DataIndex
    from opencosmo.spatial.protocols import Region

    Leaves = Mapping[tuple[str, ...], DatasetState | LightconeState]


@dataclass(frozen=True)
class SourceOps:
    """The dataset or lightcone operations a structure collection forwards."""

    filter: Callable[..., Source]
    bound: Callable[..., Source]
    sort_by: Callable[..., Source]
    take: Callable[..., Source]
    take_range: Callable[..., Source]
    take_rows: Callable[..., Source]
    select: Callable[..., Source]
    drop: Callable[..., Source]
    with_new_columns: Callable[..., Source]
    with_units: Callable[..., Source]
    get_data: Callable[..., Any]


__DATASET_OPS = SourceOps(
    filter=dsops.filter,
    bound=dsops.bound,
    sort_by=dsops.sort_by,
    take=dsops.take,
    take_range=dsops.take_range,
    take_rows=dsops.take_rows,
    select=dsops.select,
    drop=dsops.drop,
    with_new_columns=dsops.with_new_columns,
    with_units=dsops.with_units,
    get_data=dsops.get_data,
)

__LIGHTCONE_OPS = SourceOps(
    filter=lcops.filter,
    bound=lcops.bound,
    sort_by=lcops.sort_by,
    take=lcops.take,
    take_range=lcops.take_range,
    take_rows=lcops.take_rows,
    select=lcops.select,
    drop=lcops.drop,
    with_new_columns=lcops.with_new_columns,
    with_units=lcops.with_units,
    get_data=lcops.get_data,
)


def source_ops(source: Source) -> SourceOps:
    if isinstance(source, DatasetState):
        return __DATASET_OPS
    return __LIGHTCONE_OPS


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
        new_source = source_ops(state.source).filter(state.source, *masks, mode=mode)
        return __with_source_rows(state, new_source)
    if "galaxy_properties" not in state.linked:
        raise ValueError("Dataset galaxy_properties not found in collection.")

    galaxies = state.children["galaxy_properties"]
    assert not isinstance(galaxies, StructureState)
    galaxy_ops = source_ops(galaxies)
    matching = galaxy_ops.select(
        galaxy_ops.filter(galaxies, *masks, mode=mode), "fof_halo_tag"
    )
    tags = galaxy_ops.get_data(matching, "astropy", True)
    new_source = source_ops(state.source).filter(
        state.source, col("fof_halo_tag").isin(tags)
    )
    return __with_source_rows(state, new_source)


def bound(
    state: StructureState, region: Region, select_by: str | None = None
) -> StructureState:
    new_source = source_ops(state.source).bound(state.source, region, select_by)
    return __with_source_rows(state, new_source)


def sort_by(state: StructureState, column: str, invert: bool = False) -> StructureState:
    new_source = source_ops(state.source).sort_by(state.source, column, invert)
    return __with_source_rows(state, new_source)


def take(state: StructureState, n: int, at: str, mode: str) -> StructureState:
    new_source = source_ops(state.source).take(state.source, n, at, mode)
    return __with_source_rows(state, new_source)


def take_range(
    state: StructureState, start: int, end: int, mode: str
) -> StructureState:
    new_source = source_ops(state.source).take_range(state.source, start, end, mode)
    return __with_source_rows(state, new_source)


def take_rows(state: StructureState, rows: DataIndex) -> StructureState:
    new_source = source_ops(state.source).take_rows(state.source, rows)
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
            new_source = source_ops(state.source).select(
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
            new_linked[dataset] = source_ops(child).select(
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
            new_source = source_ops(state.source).drop(state.source, columns)
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
            new_linked[dataset] = source_ops(child).drop(child, columns)

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
        # Known inconsistency: this drops derived_columns.
        return replace(
            resolved,
            linked={**resolved.linked, collection_name: new_nested},
            derived_columns=frozenset(),
        )

    if dataset == state.dtype:
        new_source = source_ops(state.source).with_new_columns(
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
    # Known inconsistency: this drops derived_columns.
    return replace(
        state, source=new_source, linked=new_linked, derived_columns=frozenset()
    )


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
    return source_ops(source).with_units(
        source, convention, unit_conversions, **column_conversions
    )


def __with_source_rows(state: StructureState, new_source: Source) -> StructureState:
    linked_source = state.linked_source
    if linked_source is None:
        linked_source = state.source
    return replace(state, source=new_source, linked_source=linked_source)


def __resolved(state: StructureState) -> StructureState:
    return replace(state, linked=state.children, linked_source=None)
