from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    TypedDict,
)

from opencosmo.index import into_array

if TYPE_CHECKING:
    import numpy as np
    from astropy.table import Table

    from opencosmo import Lightcone
    from opencosmo.dataset.state import DatasetState
    from opencosmo.index import DataIndex, IndexArray


class PluginType(StrEnum):
    DatasetOpen = "dataset_open"
    DatasetInstantiate = "dataset_instantiate"
    LightconeOpen = "lightcone_open"
    LightconeInstantiate = "lightcone_instantiate"
    PostSort = "post_sort"
    IndexUpdate = "index_update"


type Verifier[T] = Callable[[T], bool]
type Plugin[T] = Callable[[T], T]


class PluginSpec[T](NamedTuple):
    plugin_type: PluginType
    verifier: Verifier[T]
    plugin: Plugin[T]


class IndexPluginSpec(NamedTuple):
    plugin_type: PluginType
    verifier: Callable[[DatasetState], bool]
    plugin: Callable[[DatasetState, DataIndex], DataIndex]


class PostSortPluginSpec[T: (DatasetState, Lightcone)](NamedTuple):
    plugin_type: PluginType
    verifier: Callable[[T], bool]
    plugin: Callable[[Table, IndexArray], dict[str, np.ndarray]]


class Plugins(TypedDict):
    dataset_open: list[PluginSpec]
    dataset_instantiate: list[PluginSpec]
    lightcone_load: list[PluginSpec]
    lightcone_instantiate: list[PluginSpec]
    index_update: list[IndexPluginSpec]


KNOWN_PLUGINS: Plugins = defaultdict(list)  # type: ignore


def register_plugin(spec: PluginSpec | IndexPluginSpec | PostSortPluginSpec) -> None:
    KNOWN_PLUGINS[str(spec.plugin_type)].append(spec)  # type: ignore


def apply_plugins[T](plugin_type: PluginType, target: T, **kwargs: Any) -> T:
    """Apply all registered plugins of the given type to target.

    kwargs are forwarded to both the verifier and plugin, used to pass
    open_kwargs through to DatasetOpen plugins.
    """
    plugins_to_apply = KNOWN_PLUGINS[str(plugin_type)]  # type: ignore
    return reduce(
        lambda t, spec: _apply_single(spec, t, **kwargs), plugins_to_apply, target
    )


def apply_index_plugins(state: DatasetState, index: DataIndex) -> DataIndex:
    """Apply all registered IndexUpdate plugins to index.

    Each plugin may expand or otherwise modify the position index returned by a
    filter or take operation. Plugins run in registration order, each seeing the
    output of the previous one.
    """
    plugins_to_apply: list[IndexPluginSpec] = KNOWN_PLUGINS[str(PluginType.IndexUpdate)]  # type: ignore
    return reduce(
        lambda idx, spec: _apply_single_index(spec, state, idx),
        plugins_to_apply,
        index,
    )


def apply_post_sort_plugins[T: (DatasetState, Lightcone)](
    state: T,
    data: Table,
    index: DataIndex,
) -> T:
    plugins_to_apply: list[PostSortPluginSpec] = KNOWN_PLUGINS[str(PluginType.PostSort)]  # type: ignore
    index_arr = into_array(index)

    return reduce(
        lambda data_, spec: _apply_single_post_sort(spec, state, data_, index_arr),
        plugins_to_apply,
        data,
    )


def _apply_single_post_sort[T: (DatasetState, Lightcone)](
    spec: PostSortPluginSpec, state: T, data: Table, index: IndexArray
):
    if spec.verifier(state):
        return spec.plugin(data, index)
    return data


def _apply_single[T](spec: PluginSpec[T], target: T, **kwargs: Any) -> T:
    if spec.verifier(target, **kwargs):
        return spec.plugin(target, **kwargs)
    return target


def _apply_single_index(
    spec: IndexPluginSpec, state: DatasetState, index: DataIndex
) -> DataIndex:
    if spec.verifier(state):
        return spec.plugin(state, index)
    return index
