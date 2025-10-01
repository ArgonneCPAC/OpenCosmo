from __future__ import annotations

from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Iterable, TypeVar

import astropy.units as u  # type: ignore
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore
from healpy import ang2vec, query_disc  # type: ignore

from opencosmo.spatial.models import (
    BoxRegionModel,
    ConeRegionModel,
    HealPixRegionModel,
)

if TYPE_CHECKING:
    from astropy.cosmology import FLRW
    from numpy.typing import NDArray

    from opencosmo.spatial.protocols import Region
    from opencosmo.units import UnitConvention
    from opencosmo.units.handler import UnitHandler

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


class ConeRegion:
    """
    Cone region for querying lightcones. Defined by RA/Dec coordinate and an angular
    size. Should always be constructed with :py:meth:`opencosmo.make_cone`

    Note that the underlying coordinate representation of the data may or may not be
    in RA and Dec. Spatial queries handle all the necessary conversions, but this may
    mean that the coordinates in the query output do not appear to match the coordinates
    of the region used to perform the query.
    """

    def __init__(self, center: SkyCoord, radius: u.Quantity):
        self.__center = center
        self.__radius = radius

    def __hash__(self):
        return hash(
            (self.__center.ra.deg, self.center.dec, self.radius.to(u.deg).value)
        )

    def into_base_convention(self, *args, **kwargs):
        return self

    def into_model(self) -> ConeRegionModel:
        return ConeRegionModel(
            center=(self.center.ra.value, self.center.dec.value),
            radius=self.__radius.value,
        )

    @property
    def center(self):
        """
        The center of this region.

        Returns
        -------
        coordinate: astropy.coordinates.SkyCoord
        """
        return self.__center

    @property
    def radius(self):
        """
        The angular radius of the region.

        Returns
        -------
        radius: astropy.units.Quantity
        """
        return self.__radius

    def contains(self, other: Any):
        """
        Check if this ConeRegion contains another ConeRegion. This region must
        be fully outside the other for this function to return True. Functionally,
        this means a region does not contain itself.

        Parameters
        ----------
        other: :py:class:`opencosmo.spatial.ConeRegion`
        """
        return self._contains(other)

    def intersects(self, other: Any):
        """
        Check if this ConeRegion intersects with another cone region. Regions which
        touch at a single point but do not cross will return False.

        Parameters
        ----------
        other: :py:class:`opencosmo.spatial.ConeRegion`
            The other region to query.

        Return
        ------
        intersects: bool
            Whether the two regions intersect
        """
        return self._intersects(other)

    def get_healpix_intersections(self, nside: int):
        phi = self.center.ra.to(u.radian).value
        dec = self.center.dec.to(u.radian).value
        # SkyCoords casts negative RAs to their positive equivalent
        # declination of north pole is +pi / 2
        # Theta of north pole is 0
        theta = np.pi / 2 - dec

        vec = ang2vec(theta, phi)
        radius = self.radius.to(u.rad).value
        return query_disc(nside, vec, radius, inclusive=True, nest=True)

    @singledispatchmethod
    def _intersects(self, other: Any):
        raise ValueError(f"Expected a 2D Sky Region but recieved {type(other)}")

    @singledispatchmethod
    def _contains(self, other: Any):
        raise ValueError(f"Expected a 2D Sky Region but recieved {type(other)}")

    @_contains.register
    def _(self, coords: SkyCoord):  # coords are assumed to be in radians
        seps = self.__center.separation(coords)
        return seps < self.__radius


@ConeRegion._contains.register  # type: ignore
def _(self, other: ConeRegion):
    dtheta = self.center.separation(other.center)
    return self.radius > (dtheta + other.radius)


@ConeRegion._intersects.register  # type: ignore
def _(self, other: ConeRegion):
    dtheta = self.center.separation(other.center)
    return dtheta < (self.radius + other.radius)


class HealPixRegion:
    def __init__(self, idxs: NDArray[np.int_], nside: int):
        self.__idxs = idxs
        self.__nside = nside

    def into_base_convention(self, *args, **kwargs):
        return self

    def into_model(self):
        return HealPixRegionModel(pixels=self.__idxs, nside=self.nside)

    @property
    def pixels(self):
        return self.__idxs

    @property
    def nside(self):
        return self.__nside

    def contains(self, other: Any):
        return False

    def intersects(self, other: Any):
        return self._intersects(other)

    @singledispatchmethod
    def _intersects(self, other: Any):
        raise ValueError(f"Expected a 2D Sky Region but recieved {type(other)}")

    @_intersects.register
    def _(self, other: ConeRegion):
        intersections = other.get_healpix_intersections(self.__nside)
        return np.any(np.isin(self.__idxs, intersections))


@HealPixRegion._intersects.register  # type: ignore
def _(self, other: HealPixRegion):
    self_pixels = self.pixels
    other_pixels = other.pixels
    return np.any(np.isin(self_pixels, other_pixels))


class FullSkyRegion:
    def __init__(self):
        pass

    def intersects(self, other: Any):
        return True

    @singledispatchmethod
    def _intersects(self, other: Any):
        raise ValueError(f"Expected a 2D Sky Region but recieved {type(other)}")

    @_intersects.register
    def _(self, other: ConeRegion):
        return True

    @_intersects.register
    def _(self, coords: SkyCoord):
        return True

    def contains(self, other: Any):
        return self._contains(other)

    @singledispatchmethod
    def _contains(self, other: Any):
        raise ValueError(f"Expected a 2D Sky Region but recieved {type(other)}")

    @_contains.register
    def _(self, other: ConeRegion):
        return True

    @_contains.register
    def _(self, coords: SkyCoord):
        return True

    def into_model(self):
        return None


@FullSkyRegion._contains.register  # type: ignore
def _(self, other: FullSkyRegion):
    return False


@FullSkyRegion._intersects.register  # type: ignore
def _(self, other: FullSkyRegion):
    return False


class BoxRegion:
    """
    A region representing a 3-dimensional box of arbitrary length, width, and depth. A
    BoxRegion can be used to query snapshot data. BoxRegions can be constructed with
    :py:meth:`opencosmo.make_box`

    When used in a spatial query, the values defining a box region will always be
    interpreted in the same unit convention as the dataset it is being used to query.
    This means two queries on the same dataset with the same box region can return
    different results if the unit convention changes between them.
    """

    def __init__(self, center: Point3d, halfwidths: BoxSize):
        self.__center = center
        self.__halfwidths = halfwidths
        self.__bounds: list[tuple[float, float]] = [
            (c - hw, c + hw) for c, hw in zip(self.__center, self.__halfwidths)
        ]

    def __repr__(self):
        return f"Box with bounds {self.bounds}"

    def into_model(self) -> BoxRegionModel:
        p1 = tuple(b[0] for b in self.bounds)
        p2 = tuple(b[1] for b in self.bounds)
        return BoxRegionModel(p1=p1, p2=p2)

    def bounding_box(self) -> BoxRegion:
        return self

    def into_base_convention(
        self,
        unit_handler: UnitHandler,
        columns: Iterable[str],
        from_: UnitConvention,
        unit_kwargs: dict[str, Any] = {},
    ):
        center = {col: dim for col, dim in zip(columns, self.__center)}
        halfwidth = {col: dim for col, dim in zip(columns, self.__halfwidths)}

        new_center = tuple(
            v.value
            for v in unit_handler.into_base_convention(center, unit_kwargs).values()
        )
        new_halfwidth = tuple(
            v.value
            for v in unit_handler.into_base_convention(halfwidth, unit_kwargs).values()
        )

        return BoxRegion(new_center, new_halfwidth)

    @property
    def bounds(self):
        """
        The bounds of this region in the form [(min,max) ...]

        Returns
        -------
        bounds: list[tuple(float,float), ....]
        """
        return self.__bounds

    def contains(self, other: Any) -> bool | np.ndarray:
        """
        Check if this box region contains another region or set of points.
        Points should be passed in as a numpy array with shape (3, n_points). Note
        that this method requires that the bounds of the test region be fully inside the
        other region. Functionally this means that a region does not contain itself.

        Parameters
        ----------
        other: BoxRegion | np.ndarray
            The region or points to check

        Returns
        -------
        contained: bool or np.ndarray[bool]
            Whether the region or points are contained in the region

        Raises
        ------
        ValueError
            If the input not a region or points
        """
        return self._contains(other)

    @singledispatchmethod
    def _contains(self, other):
        raise ValueError(f"Expected a 3D region, got {type(other)}")

    @_contains.register
    def _(self, coords: np.ndarray):
        if coords.shape[0] != 3:
            raise ValueError("Expected a coordinate array!")

        mask = np.ones(coords.shape[1], dtype=bool)
        for bound, col in zip(self.bounds, coords):
            mask &= (col > bound[0]) & (col < bound[1])

        return mask

    def intersects(self, other: Region) -> bool:
        """
        Check if this BoxRegion intersects another BoxRegion. This function
        returns true of the regions intersect in any way, including if one
        contains the other. Regions that share a bound but do not cross
        will return False.

        Parameters
        ----------
        other: BoxRegion
            The region to check

        Returns
        -------
        intersects: bool
            Whether the regions intersect

        Raises
        ------
        ValueError:
            If the input is not a BoxRegion

        """
        if not isinstance(other, BoxRegion):
            raise ValueError(f"Expected a 3D region, but got {type(other)}")

        for b1, b2 in zip(self.bounds, other.bounds):
            if b1[0] > b2[1] or b1[1] < b2[0]:
                return False
        return True


@BoxRegion._contains.register  # type: ignore
def _(self, other: BoxRegion):
    for b1, b2 in zip(self.bounds, other.bounds):
        if b1[0] > b2[0] or b1[1] < b2[1]:
            return False
    return True
