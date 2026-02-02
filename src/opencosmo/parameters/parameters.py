from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type, Union, get_origin

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
        header_group = header[header_path]
        header_data = dict(header_group.attrs)
    except KeyError:
        return parameter_model()  # Defaults are possible

    for key, value in header_group.values():
        if not isinstance(value, h5py.Dataset):
            continue
        header_data[key] = value[:]

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


def should_dump_to_hdf5_dataset(obj: Any):
    if isinstance(obj, np.ndarray):
        return True
    if isinstance(obj, list) and isinstance(obj[0], (int, float)):
        return True
    return False


def write_header_attributes(file: h5py.File, header_path: str, parameters: BaseModel):
    group = file.require_group(f"header/{header_path}")
    pars = parameters.model_dump(by_alias=True)
    array_pars = {
        key: val for key, val in pars.items() if should_dump_to_hdf5_dataset(val)
    }

    for key, value in pars.items():
        if key in array_pars:
            continue
        if value is None:
            group.attrs[key] = ""
        else:
            group.attrs[key] = value

    for par_name, data in array_pars.items():
        group.create_dataset(name=par_name, data=data)

    return None


def get_empty_from_optional(type_: Type):
    origin = get_origin(type_)
    if origin is Union:
        np_equivalent = np.dtype(type_.__args__[0])
        return h5py.Empty(np_equivalent)
