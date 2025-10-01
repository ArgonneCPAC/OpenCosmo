from __future__ import annotations

from typing import TYPE_CHECKING, cast

import astropy.units as u  # type: ignore
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore

from opencosmo.spatial.models import (
    BoxRegionModel,
    ConeRegionModel,
    HealPixRegionModel,
)
from opencosmo.spatial.region import (
    BoxRegion,
    ConeRegion,
    HealPixRegion,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from opencosmo.spatial.region import (
        BoxSize,
        Point2d,
        Point3d,
    )


def from_model(model: BaseModel):
    match model:
        case ConeRegionModel():
            return make_cone(model.center, model.radius)
        case BoxRegionModel():
            return make_box(model.p1, model.p2)
        case HealPixRegionModel():
            return HealPixRegion(np.array(model.pixels), model.nside)
        case _:
            raise ValueError(f"Invalid region model type {type(model)}")


def make_box(p1: Point3d, p2: Point3d):
    """
    Create a 3-Dimensional box region of arbitrary size.
    The quantities of the box region are unitless, but will be converted
    to the unit convention of any dataset they interact with.

    Parameters
    ----------
    p1: (float, float, float)
        3D Point definining one corner of the box
    p1: (float, float, float)
        3D Point definining the other corner of the box

    Returns
    -------
    region: :py:class:`opencosmo.spatial.BoxRegion`
        The constructed region


    Raises
    ------
    ValueError
        If the region has zero length in any dimension
    """
    if len(p1) != 3 or len(p2) != 3:
        raise ValueError("Expected two 3-dimensional points")
    bl = tuple(min(p1d, p2d) for p1d, p2d in zip(p1, p2))
    tr = tuple(max(p1d, p2d) for p1d, p2d in zip(p1, p2))

    width = tuple(trd - bld for bld, trd in zip(bl, tr))
    center = tuple(bld + wd / 2.0 for bld, wd in zip(bl, width))
    if any(w == 0 for w in width):
        raise ValueError("At least one dimension of this box has zero length!")

    if isinstance(width, float) or isinstance(width, int):
        width = (width, width, width)

    halfwidth = cast("BoxSize", tuple(float(w / 2) for w in width))
    center = cast("Point3d", center)

    return BoxRegion(center, halfwidth)


def make_cone(center: Point2d | SkyCoord, radius: float | u.Quantity):
    """
    Construct a cone region used for querying lightcones. A cone
    region is defined by a location on the sky and an angular size,
    which is used as a radius.

    Parameters
    ----------
    center: astropy.coordinates.SkyCord | tuple[float, float]
        The center of the cone region. If a unitless tuple is passed,
        the values are assumed to be in degrees.

    radius: astropy.units.Quantity | float
        The radius of the region. If a unitless value is passed,
        it is assumed to be in degrees.

    Returns
    -------
    region: :py:class:`opencosmo.spatial.ConeRegion`
        The constructed region
    """
    coord: SkyCoord
    match center:
        case SkyCoord():
            coord = center
        case (float(ra) | int(ra), float(dec) | int(dec)):
            coord = SkyCoord(ra * u.deg, dec * u.deg)
        case (u.Quantity(), u.Quantity()):
            coord = SkyCoord(*center)
        case _:
            raise ValueError("Invalid center for Cone region")
    if isinstance(radius, (float, int)):
        radius = radius * u.deg
    return ConeRegion(coord, radius)
