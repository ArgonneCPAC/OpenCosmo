from typing import Protocol, Iterable
from opencosmo.parameters import SimulationParameters


class SpatialIndex(Protocol):
    def __init__(self, simulation_parameters: SimulationParameters, max_level: int): ...

    def get_regions(self, *args, **kwargs) -> Iterable[int]: ...   


