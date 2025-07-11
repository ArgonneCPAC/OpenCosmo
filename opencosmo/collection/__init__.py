from .io import get_collection_type, open_simulation_files
from .lightcone import Lightcone, open_lightcone
from .protocols import Collection
from .simulation import SimulationCollection
from .structure import StructureCollection, open_linked_file, open_linked_files

__all__ = [
    "Collection",
    "SimulationCollection",
    "StructureCollection",
    "open_multi_dataset_file",
    "SimulationCollection",
    "open_simulation_files",
    "open_linked_file",
    "open_linked_files",
    "Lightcone",
    "open_lightcone",
    "get_collection_type",
]
