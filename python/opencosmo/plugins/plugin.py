from collections import defaultdict
from enum import StrEnum
from functools import reduce
from typing import Callable, NamedTuple, TypedDict

import opencosmo as oc


class PluginType(StrEnum):
    DatasetOpen = "dataset_open"
    DatasetInstantiate = "dataset_instantiate"
    LightconeLoad = "lightcone_load"
    LightconeInstantiate = "lightcone_instantiate"


DatasetTransformationPlugin = Callable[[oc.Dataset], oc.Dataset]
LightconeTransformationPlugn = Callable[[oc.Lightcone], oc.Lightcone]


type DatasetType[T: (oc.Dataset, oc.Lightcone)] = T
type Verifier[T: DatasetType] = Callable[[T], bool]
type Plugin[T: DatasetType] = Callable[[T], T]


class PluginSpec[T: DatasetType](NamedTuple):
    plugin_type: PluginType
    verifier: Verifier[T]
    plugin: Plugin[T]


class Plugins(TypedDict):
    dataset_open: list[PluginSpec[oc.Dataset]]
    dataset_instantiate: list[PluginSpec[oc.Dataset]]
    lightcone_load: list[PluginSpec[oc.Lightcone]]
    lightcone_instantiate: list[PluginSpec[oc.Lightcone]]


KNOWN_PLUGINS: Plugins = defaultdict(list)  # type: ignore


def register_plugin[T: DatasetType](
    plugin_type: PluginType,
    verifier: Verifier[T],
    plugin: Plugin[T],
) -> None:
    spec = PluginSpec(plugin_type=plugin_type, verifier=verifier, plugin=plugin)
    KNOWN_PLUGINS[str(plugin_type)].append(spec)  # type: ignore


def apply_plugins[T: DatasetType](plugin_type: PluginType, dataset: T) -> T:
    plugins_to_apply = KNOWN_PLUGINS[str(plugin_type)]  # type: ignore
    return reduce(
        lambda ds, spec: apply_single_plugin(spec, ds), plugins_to_apply, dataset
    )


def apply_single_plugin[T: DatasetType](spec: PluginSpec[T], dataset: T) -> T:
    if spec.verifier(dataset):
        return spec.plugin(dataset)
    return dataset
