from __future__ import annotations

from typing import Protocol

import numpy as np

Point3d = tuple[float, float, float]


class Region(Protocol):
    def __init__(self, *args, **kwargs): ...
    def contains(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray: ...
    def bounding_box(self) -> BoxRegion: ...


class BoxRegion:
    def __init__(self, p1: Point3d, p2: Point3d):
        self.p1 = tuple([min(p1[0], p2[0]), min(p1[1], p2[1]), min(p1[2], p2[2])])
        self.p2 = tuple([max(p1[0], p2[0]), max(p1[1], p2[1]), max(p1[2], p2[2])])

    def bounding_box(self) -> BoxRegion:
        return self

    def contains(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (
            (x >= self.p1[0])
            & (x <= self.p2[0])
            & (y >= self.p1[1])
            & (y <= self.p2[1])
            & (z >= self.p1[2])
            & (z <= self.p2[2])
        )
