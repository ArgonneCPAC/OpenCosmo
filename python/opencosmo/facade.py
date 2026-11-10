from __future__ import annotations

from typing import TYPE_CHECKING, overload

from opencosmo.collection.healpix_map import healpix_map as hm
from opencosmo.collection.healpix_map.state import HealpixMapState
from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.particles import particles as pc
from opencosmo.collection.particles.state import ParticleState
from opencosmo.collection.simulation import simulation as sim
from opencosmo.collection.simulation.state import SimulationState
from opencosmo.collection.structure import structure as sc
from opencosmo.collection.structure.state import StructureState
from opencosmo.dataset import dataset as ds
from opencosmo.dataset.state import DatasetState
from opencosmo.power_spectrum import power_spectrum as ps
from opencosmo.power_spectrum.state import PowerSpectrumState

if TYPE_CHECKING:
    from opencosmo.collection.healpix_map.healpix_map import HealpixMap
    from opencosmo.collection.lightcone.lightcone import Lightcone
    from opencosmo.collection.particles.particles import ParticleCollection
    from opencosmo.collection.simulation.simulation import SimulationCollection
    from opencosmo.collection.structure.structure import StructureCollection
    from opencosmo.dataset.dataset import Dataset
    from opencosmo.power_spectrum.power_spectrum import PowerSpectrum


@overload
def wrap(state: DatasetState) -> Dataset: ...
@overload
def wrap(state: LightconeState) -> Lightcone: ...
@overload
def wrap(state: StructureState) -> StructureCollection: ...
@overload
def wrap(state: SimulationState) -> SimulationCollection: ...
@overload
def wrap(state: HealpixMapState) -> HealpixMap: ...
@overload
def wrap(state: ParticleState) -> ParticleCollection: ...
@overload
def wrap(state: PowerSpectrumState) -> PowerSpectrum: ...
def wrap(
    state: DatasetState
    | LightconeState
    | StructureState
    | SimulationState
    | HealpixMapState
    | ParticleState
    | PowerSpectrumState,
) -> (
    Dataset
    | Lightcone
    | StructureCollection
    | SimulationCollection
    | HealpixMap
    | ParticleCollection
    | PowerSpectrum
):
    """Wrap a backend state in the facade that presents it to the user."""
    match state:
        case DatasetState():
            return ds.Dataset(state)
        case LightconeState():
            return lc.Lightcone(state)
        case StructureState():
            return sc.StructureCollection(state)
        case SimulationState():
            return sim.SimulationCollection(state)
        case HealpixMapState():
            return hm.HealpixMap(state)
        case ParticleState():
            return pc.ParticleCollection(state)
        case PowerSpectrumState():
            return ps.PowerSpectrum(state)
    raise TypeError(f"Cannot wrap object of type {type(state).__name__}")
