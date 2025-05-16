from __future__ import annotations

from functools import partial, singledispatchmethod
from typing import Any, Protocol

import numpy as np
from astropy.cosmology import FLRW  # type: ignore

from opencosmo.transformations.units import UnitConvention

Point3d = tuple[float, float, float]
BoxSize = tuple[float, float, float]


class Region(Protocol):
    def contains(self, other: Any) -> bool | np.ndarray: ...

    def bounding_box(self) -> BoxRegion: ...
    def into_scalefree(self, from_: UnitConvention, cosmology: FLRW, z: float): ...


def physical_to_scalefree(value: float, cosmology: FLRW, z: float):
    a = cosmology.scale_factor(z)
    comoving_value = value / a
    return comoving_to_scalefree(comoving_value, cosmology)


def comoving_to_scalefree(value: float, cosmology: FLRW):
    h = cosmology.h
    scalefree_value = value * h
    return scalefree_value


class BoxRegion:
    def __init__(self, center: Point3d, halfwidths: BoxSize):
        self.center = center
        self.halfwidths = halfwidths
        self.bounds: list[tuple[float, float]] = [
            (c - hw, c + hw) for c, hw in zip(self.center, self.halfwidths)
        ]

    def __repr__(self):
        return (
            f"Box centered at {self.center} with widths "
            f"{tuple(2 * hw for hw in self.halfwidths)}"
        )

    def bounding_box(self) -> BoxRegion:
        return self

    def into_scalefree(self, from_: UnitConvention, cosmology: FLRW, z: float):
        match from_:
            case UnitConvention.SCALEFREE | UnitConvention.UNITLESS:
                return self
            case UnitConvention.COMOVING:
                fn = partial(comoving_to_scalefree, cosmology=cosmology)
            case UnitConvention.PHYSICAL:
                fn = partial(physical_to_scalefree, cosmology=cosmology, z=z)
        new_center = tuple(fn(dim) for dim in self.center)
        new_halfwidth = tuple(fn(dim) for dim in self.halfwidths)
        return BoxRegion(new_center, new_halfwidth)

    def contains(self, other: Any) -> bool | np.ndarray:
        return self.__contains(other)

    @singledispatchmethod
    def __contains(self, other: BoxRegion):
        for b1, b2 in zip(self.bounds, other.bounds):
            if b1[0] > b2[0] or b1[1] < b2[1]:
                return False
        return True

    @__contains.register
    def _(self, coords: np.ndarray):
        if coords.shape[0] != 3:
            raise ValueError("Expected a coordinate array!")

        mask = np.ones(coords.shape[1], dtype=bool)
        for bound, col in zip(self.bounds, coords):
            mask &= (col > bound[0]) & (col < bound[1])

        return mask

    def intersects(self, other: BoxRegion) -> bool:
        for b1, b2 in zip(self.bounds, other.bounds):
            if b1[0] > b2[1] or b1[1] < b2[0]:
                return False
        return True
