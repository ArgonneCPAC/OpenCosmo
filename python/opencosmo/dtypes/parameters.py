from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from pydantic import ValidationError

if TYPE_CHECKING:
    from pydantic import BaseModel


def read_header_attributes(
    file: h5py.File | h5py.Group,
    header_path: str,
    parameter_model: type[BaseModel],
    **kwargs,
):
    header = file["header"]
    try:
        header_group = header[header_path]
        header_data = dict(header_group.attrs)
    except KeyError:
        return parameter_model()  # Defaults are possible

    for key, value in header_group.items():
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
    return bool(isinstance(obj, list) and isinstance(obj[0], (int, float)))


def write_header_attributes(file: h5py.File, header_path: str, parameters: BaseModel):
    group = file.require_group(f"header/{header_path}")
    pars = parameters.model_dump(by_alias=True, exclude_none=True)
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
