from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from functools import reduce
from typing import Callable, NamedTuple, TypedDict

from opencosmo import dataset as ds
from opencosmo.collection.lightcone import lightcone as lc


class PluginType(StrEnum):
    DatasetOpen = "dataset_open"
    DatasetInstantiate = "dataset_instantiate"
    LightconeLoad = "lightcone_load"
    LightconeInstantiate = "lightcone_instantiate"


DatasetTransformationPlugin = Callable[[ds.Dataset], ds.Dataset]
LightconeTransformationPlugin = Callable[[lc.Lightcone], dict[str, ds.Dataset]]

type Verifier[T: (ds.Dataset, lc.Lightcone, ds.state.DatasetState)] = Callable[
    [T], bool
]
type Plugin[T: (ds.Dataset, lc.Lightcone, ds.state.DatasetState)] = Callable[[T], T]


class PluginSpec[T: (ds.Dataset, lc.Lightcone, ds.state.DatasetState)](NamedTuple):
    plugin_type: PluginType
    verifier: Verifier[T]
    plugin: Plugin[T]


class Plugins(TypedDict):
    dataset_open: list[PluginSpec[ds.Dataset]]
    dataset_instantiate: list[PluginSpec[ds.state.DatasetState]]
    lightcone_load: list[PluginSpec[lc.Lightcone]]
    lightcone_instantiate: list[PluginSpec[lc.Lightcone]]


KNOWN_PLUGINS: Plugins = defaultdict(list)  # type: ignore


def register_plugin[T: (ds.Dataset, lc.Lightcone, ds.state.DatasetState)](
    plugin_type: PluginType,
    verifier: Verifier[T],
    plugin: Plugin[T],
) -> None:
    spec = PluginSpec(plugin_type=plugin_type, verifier=verifier, plugin=plugin)
    KNOWN_PLUGINS[str(plugin_type)].append(spec)  # type: ignore


def apply_plugins[T: (ds.Dataset, lc.Lightcone, ds.state.DatasetState)](
    plugin_type: PluginType, dataset: T
) -> T:
    plugins_to_apply = KNOWN_PLUGINS[str(plugin_type)]  # type: ignore
    return reduce(
        lambda ds_, spec: apply_single_plugin(spec, ds_), plugins_to_apply, dataset
    )


def apply_single_plugin[T: (ds.Dataset, lc.Lightcone, ds.state.DatasetState)](
    spec: PluginSpec[T], dataset: T
) -> T:
    if spec.verifier(dataset):
        return spec.plugin(dataset)
    return dataset
