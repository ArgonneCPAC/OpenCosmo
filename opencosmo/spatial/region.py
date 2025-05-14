from __future__ import annotations

from typing import Protocol

import numpy as np

Point3d = tuple[float, float, float]


class Region(Protocol):
    def __init__(self, *args, **kwargs): ...
    def contains(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray: ...
    def bounding_box(self) -> BoxRegion: ...


class BoxRegion:
    def __init__(self, center: Point3d, halfwidth: float):
        self.center = center
        self.halfwidth = halfwidth
        self.bounds = [(c - self.halfwidth, c + self.halfwidth) for c in self.center]

    def __repr__(self):
        return f"center: {self.center}, width: {self.halfwidth * 2}"

    def bounding_box(self) -> BoxRegion:
        return self

    def contains(self, other: BoxRegion):
        for b1, b2 in zip(self.bounds, other.bounds):
            if b1[0] > b2[0] or b1[1] < b2[1]:
                return False
        return True

    def intersects(self, other: BoxRegion) -> bool:
        for b1, b2 in zip(self.bounds, other.bounds):
            if b1[0] > b2[1] or b1[1] < b2[0]:
                return False
        return True
