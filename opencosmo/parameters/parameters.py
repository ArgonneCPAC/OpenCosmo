from typing import Type, Union, get_origin

import h5py
import numpy as np
from pydantic import BaseModel, ValidationError


def read_header_attributes(
    file: h5py.File, header_path: str, parameter_model: Type[BaseModel], **kwargs
):
    header_data = file["header"][header_path].attrs
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
        if isinstance(value, dict):
            write_header_attributes(
                file, f"{header_path}/{key}", getattr(parameters, key)
            )
            continue
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
