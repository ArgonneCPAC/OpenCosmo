from typing import Type

import h5py
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
