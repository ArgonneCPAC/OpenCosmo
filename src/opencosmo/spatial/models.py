from pydantic import BaseModel


class BoxRegionModel(BaseModel):
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]


class ConeRegionModel(BaseModel):
    center: tuple[float, float]
    radius: float


class HealPixRegionModel(BaseModel):
    pixels: list[int]
    nside: int


RegionModel = BoxRegionModel | ConeRegionModel | HealPixRegionModel
