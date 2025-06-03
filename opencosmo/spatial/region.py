from __future__ import annotations

from functools import partial, singledispatchmethod
from typing import Any, TypeVar

import astropy.units as u  # type: ignore
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore
from astropy.cosmology import FLRW  # type: ignore

from opencosmo.parameters.file import BoxRegionModel, ConeRegionModel
from opencosmo.transformations.units import UnitConvention

T = TypeVar("T", float, u.Quantity)

Point3d = tuple[float, float, float]
Point2d = tuple[T, T]
BoxSize = tuple[float, float, float]


def physical_to_scalefree(value: float, cosmology: FLRW, z: float):
    a = cosmology.scale_factor(z)
    comoving_value = value / a
    return comoving_to_scalefree(comoving_value, cosmology)


def comoving_to_scalefree(value: float, cosmology: FLRW):
    h = cosmology.h
    scalefree_value = value * h
    return scalefree_value


class FullSkyRegion:
    def __init__(self):
        pass

    def intersects(self, other: Any):
        return True

    def contains(self, other: Any):
        return True


class ConeRegion:
    """
    Cone region for querying lightcones.

    Raw data: -pi < phi < pi, 0 < theta < pi
    RA/Dec: 0 < RA < 2*pi, -pi < dec < pi


    """

    def __init__(self, center: SkyCoord, radius: u.Quantity):
        self.__center = center
        self.__radius = radius

    def into_scalefree(self, *args, **kwargs):
        return self

    def into_model(self) -> ConeRegionModel:
        return ConeRegionModel(
            center=(self.center.ra.value, self.center.dec.value),
            radius=self.__radius.value,
        )

    @property
    def center(self):
        return self.__center

    @property
    def radius(self):
        return self.__radius

    def to_healpix(self):
        pass

    def intersects(self, other: ConeRegion):
        dtheta = self.center.separation(other.center)
        return dtheta < (self.__radius + other.__radius)

    def contains(self, other: Any):
        return self.__contains(other)

    @singledispatchmethod
    def __contains(self, other: ConeRegion):
        dtheta = self.__center.separation(other.center)
        return self.__radius > (dtheta + other.__radius)

    @__contains.register
    def _(self, coords: SkyCoord):  # coords are assumed to be in radians
        seps = self.__center.separation(coords)
        return seps < self.__radius


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

    def into_model(self) -> BoxRegionModel:
        p1 = (b[0] for b in self.bounds)
        p2 = (b[1] for b in self.bounds)
        return BoxRegionModel(p1=p1, p2=p2)

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
