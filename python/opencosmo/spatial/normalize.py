from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SpatialNormalizationContext:
    """Dataset-specific metadata needed to normalize a spatial query."""

    dimensions: Literal[2, 3]
    coordinate_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dimensions not in (2, 3):
            raise ValueError("spatial dimensions must be 2 or 3")
        coordinate_names = tuple(self.coordinate_names)
        if len(coordinate_names) != self.dimensions:
            raise ValueError("coordinate name count must match spatial dimensions")
        if not all(isinstance(name, str) and name for name in coordinate_names):
            raise ValueError("coordinate names must be nonempty strings")
        object.__setattr__(self, "coordinate_names", coordinate_names)
