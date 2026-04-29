from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, TypedDict

from opencosmo.index import into_array

if TYPE_CHECKING:
    import h5py
    import numpy as np
    from astropy.table import Table
    from mpi4py import MPI

    from opencosmo import Lightcone
    from opencosmo.dataset.state import DatasetState
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex, IndexArray
    from opencosmo.spatial.tree import Tree, TreePartition


class PluginType(StrEnum):
    DatasetOpen = "dataset_open"
    DatasetInstantiate = "dataset_instantiate"
    LightconeOpen = "lightcone_open"
    LightconeInstantiate = "lightcone_instantiate"
    PostSort = "post_sort"
    IndexUpdate = "index_update"
    Partition = "partition"


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


class PartitionPluginSpec(NamedTuple):
    plugin_type: PluginType
    verifier: Callable[[OpenCosmoHeader, h5py.Group, h5py.Group], bool]
    plugin: Callable[
        [MPI.Comm, h5py.Group, h5py.Group, Optional[Tree], Optional[int]],
        Optional[TreePartition],
    ]


class Plugins(TypedDict):
    dataset_open: list[PluginSpec]
    dataset_instantiate: list[PluginSpec]
    lightcone_load: list[PluginSpec]
    lightcone_instantiate: list[PluginSpec]
    index_update: list[IndexPluginSpec]
    partition: list[PartitionPluginSpec]


KNOWN_PLUGINS: Plugins = defaultdict(list)  # type: ignore


def register_plugin(
    spec: PluginSpec | IndexPluginSpec | PostSortPluginSpec | PartitionPluginSpec,
) -> None:
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


def apply_partition_plugins(
    comm: MPI.Comm,
    header: OpenCosmoHeader,
    index_group: h5py.Group,
    data_group: h5py.Group,
    tree: Optional[Tree] = None,
    min_level: Optional[int] = None,
) -> Optional[TreePartition]:
    partition_plugins = KNOWN_PLUGINS[str(PluginType.Partition)]  # type: ignore
    if len(partition_plugins) == 0:
        return None
    if len(partition_plugins) > 1:
        raise ValueError("Only one partition plugin is allowed at a time")
    plugin_spec = partition_plugins[0]
    if not plugin_spec.verifier(header, index_group, data_group):
        return None

    return plugin_spec.plugin(comm, index_group, data_group, tree, min_level)


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
