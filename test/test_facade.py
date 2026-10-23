import opencosmo as oc
import pytest
from opencosmo.collection.healpix_map.state import HealpixMapState
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.simulation.state import SimulationState
from opencosmo.collection.structure.state import StructureState
from opencosmo.dataset.state import DatasetState
from opencosmo.facade import wrap
from opencosmo.io.io import open_state

CASES = {
    "dataset": (
        lambda data: [data.snapshot.primary.halo_properties],
        DatasetState,
        oc.Dataset,
    ),
    "lightcone": (
        lambda data: [
            data.lightcone.step(600).halo_properties,
            data.lightcone.step(601).halo_properties,
        ],
        LightconeState,
        oc.Lightcone,
    ),
    "structure": (
        lambda data: data.snapshot.primary.halos,
        StructureState,
        oc.StructureCollection,
    ),
    "simulation": (
        lambda data: [data.snapshot.multi_simulation],
        SimulationState,
        oc.SimulationCollection,
    ),
    "healpix_map": (
        lambda data: [data.healpix_map],
        HealpixMapState,
        oc.HealpixMap,
    ),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_open_wraps_opened_state(test_data, case):
    paths, state_type, facade_type = case
    state = open_state(*paths(test_data))

    assert isinstance(state, state_type)
    assert isinstance(wrap(state), facade_type)
    assert isinstance(oc.open(*paths(test_data)), facade_type)


def test_wrap_rejects_non_state():
    with pytest.raises(TypeError, match="Cannot wrap object of type int"):
        wrap(1)  # type: ignore[call-overload]


def test_open_state_rejects_invalid_mpi_mode(test_data):
    with pytest.raises(ValueError, match="Invalid mpi_mode 'bogus'"):
        open_state(test_data.snapshot.primary.halo_properties, mpi_mode="bogus")
