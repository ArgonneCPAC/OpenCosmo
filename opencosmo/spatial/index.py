from typing import Protocol

from opencosmo.parameters import SimulationParameters


class SpatialIndex(Protocol):
    def __init__(self, simulation_parameters: SimulationParameters, max_level: int): ...
