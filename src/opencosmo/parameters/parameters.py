from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union, get_origin

import h5py
import numpy as np
from pydantic import ValidationError

if TYPE_CHECKING:
    from pydantic import BaseModel


def read_header_attributes(
    file: h5py.File | h5py.Group,
    header_path: str,
    parameter_model: Type[BaseModel],
    **kwargs,
):
    header = file["header"]
    try:
        header_data = header[header_path].attrs
    except KeyError:
        return parameter_model()  # Defaults are possible
    try:
        parameters = parameter_model(**header_data, **kwargs)
    except ValidationError as e:
        msg = (
            "\nHeader attributes do not match the expected format for "
            f"{parameter_model.__name__}. "
            "Are you sure this is an OpenCosmo file?\n"
        )
        raise ValidationError.from_exception_data(msg, e.errors())  # type: ignore
    return parameters


def write_header_attributes(file: h5py.File, header_path: str, parameters: BaseModel):
    group = file.require_group(f"header/{header_path}")
    pars = parameters.model_dump(by_alias=True)
    for key, value in pars.items():
        if value is None:
            group.attrs[key] = ""
        else:
            group.attrs[key] = value

    return None


def get_empty_from_optional(type_: Type):
    origin = get_origin(type_)
    if origin is Union:
        np_equivalent = np.dtype(type_.__args__[0])
        return h5py.Empty(np_equivalent)
