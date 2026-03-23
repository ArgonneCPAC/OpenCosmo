from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from .utils import empty_string_to_none


class LightconeParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "lightcone"
    z_range: Optional[tuple[float, float]] = None

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            return {k: empty_string_to_none(v) for k, v in data.items()}
        return data
