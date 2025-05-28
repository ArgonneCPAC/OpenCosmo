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
            hw1, hw2, hw3 = model.halfwidth
            return Box(model.center, (2 * hw1, 2 * hw2, 2 * hw3))
        case _:
            raise ValueError(f"Invalid region model type {type(model)}")


def Box(center: Point3d, width: float | BoxSize):
    if isinstance(width, float) or isinstance(width, int):
        width = (width, width, width)

    halfwidth = cast(BoxSize, tuple(float(w / 2) for w in width))
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
