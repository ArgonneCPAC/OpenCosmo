from typing import cast

import astropy.units as u  # type: ignore
from astropy.coordinates import SkyCoord  # type: ignore
from pydantic import BaseModel

from opencosmo.parameters.file import BoxRegionModel, ConeRegionModel
from opencosmo.spatial.region import BoxRegion, BoxSize, ConeRegion, Point2d, Point3d


def from_model(model: BaseModel):
    match model:
        case ConeRegionModel():
            return Cone(model.center, model.radius)
        case BoxRegionModel():
            return Box(model.p1, model.p2)
        case _:
            raise ValueError(f"Invalid region model type {type(model)}")


def Box(p1: Point3d, p2: Point3d):
    """
    Create a 3-Dimensional box region of arbitrary size.
    The quantities of the box region are unitless, but will be converted
    to the unit convention of any dataset they interact with.

    Parameters:
    -----------
    p1: (float, float, float)
        3D Point definining one corner of the box
    p1: (float, float, float)
        3D Point definining the other corner of the box

    Returns:
    --------
    region: BoxRegion
        The constructed region


    Raises:
    -------
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

    halfwidth = cast(BoxSize, tuple(float(w / 2) for w in width))
    center = cast(Point3d, center)

    return BoxRegion(center, halfwidth)


def Cone(center: Point2d | SkyCoord, radius: float | u.Quantity):
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
    if isinstance(radius, float):
        radius = radius * u.deg
    return ConeRegion(coord, radius)
