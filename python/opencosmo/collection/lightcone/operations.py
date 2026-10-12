from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import healpy as hp
import numpy as np

from opencosmo.collection.lightcone import state as lcst
from opencosmo.column import col
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.take import (
    get_end_take_index,
    get_random_take_index,
    get_range_take_index,
    get_rows_take_index,
)
from opencosmo.index import empty, get_range, into_array, rebuild_by_ranges
from opencosmo.spatial.index import project_on_index

if TYPE_CHECKING:
    from collections.abc import Callable

    import astropy.units as u
    import numpy.typing as npt

    from opencosmo.collection.lightcone.healpix_map import HealpixMap
    from opencosmo.collection.lightcone.state import LeafKey, LightconeState
    from opencosmo.column.column import ColumnMask, CompoundColumnMask
    from opencosmo.dataset.state import DatasetState
    from opencosmo.index import DataIndex
    from opencosmo.spatial.protocols import Region

    Leaves = tuple[tuple[LeafKey, DatasetState], ...]


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
    return __map_leaves(state, lambda leaf: dsops.filter(leaf, *masks, mode=mode))


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


def __prune_empty(leaves: Leaves, keep_all_empty: bool = True) -> Leaves:
    """
    Drop empty groups at every key level. If every group at a level is empty,
    keep all of them (``keep_all_empty``) or only the first.
    """
    if not leaves[0][0]:
        return leaves
    pruned = {
        step: __prune_empty(group, keep_all_empty)
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
