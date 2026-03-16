import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer


class BoxRegionModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]


class SkyboxRegionModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    p1: tuple[float, float]
    p2: tuple[float, float]


class ConeRegionModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    center: tuple[float, float]
    radius: float


class HealpixRegionModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    pixels: frozenset[int]
    nside: int

    @field_serializer("pixels")
    def serialize_pixels(self, value):
        pixels = np.array(list(value))
        return list(np.sort(pixels))


RegionModel = BoxRegionModel | ConeRegionModel | HealpixRegionModel | SkyboxRegionModel
