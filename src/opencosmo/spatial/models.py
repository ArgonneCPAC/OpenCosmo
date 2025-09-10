from pydantic import BaseModel, ConfigDict, field_serializer


class BoxRegionModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]


class ConeRegionModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    center: tuple[float, float]
    radius: float


class HealPixRegionModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    pixels: frozenset[int]
    nside: int

    @field_serializer("pixels")
    def serialize_pixels(self, value):
        return list(value)


RegionModel = BoxRegionModel | ConeRegionModel | HealPixRegionModel
