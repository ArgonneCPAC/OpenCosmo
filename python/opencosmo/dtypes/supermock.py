from typing import Any, Callable, ClassVar

import astropy.units as u
import numpy as np
from pydantic import BaseModel, ConfigDict, model_serializer

from .units import register_units


def to_numpy(parameters: "SuperMockParams") -> dict[str, Any]:
    output = {}
    for name, value in parameters.dict().items():
        if isinstance(value, list):
            value = np.array(value)
        output[name] = value

    return output


class SuperMockParams(BaseModel):
    ACCESS_PATH: ClassVar[str] = "supermock"
    ACCESS_TRANSFORMATION: ClassVar[Callable] = to_numpy
    model_config = ConfigDict(frozen=True)
    mah_age: tuple[float, ...]
    mah_step: tuple[int, ...]
    mah_redshift: tuple[float, ...]
    sfh_age: tuple[float, ...]
    sfh_redshift: tuple[float, ...]

    @model_serializer(mode="wrap")
    def serialize_and_add_metadata(self, handler) -> dict[str, Any]:
        # 1. Let Pydantic serialize the model into a standard dict
        default_dict = handler(self)

        # 2. Modify or add extra metadata to the output dict
        for name, val in default_dict.items():
            if isinstance(val, tuple):
                default_dict[name] = list(val)

        return default_dict


register_units(SuperMockParams, "mah_age", u.Gyr)
register_units(SuperMockParams, "sfh_age", u.Gyr)
