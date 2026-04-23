from __future__ import annotations

from typing import TYPE_CHECKING

from opencosmo.dtypes import hacc, lightcone
from opencosmo.dtypes.diffsky import offset_top_host_idx, top_host_idx

if TYPE_CHECKING:
    from pydantic import BaseModel

    from opencosmo.dtypes.file import FileParameters


def get_dtype_parameters(
    file_parameters: FileParameters,
) -> dict[str, dict[str, type[BaseModel]]]:
    if file_parameters.origin == "HACC":
        known_dtype_params = hacc.DATATYPE_PARAMETERS
    else:
        known_dtype_params = {}
    dtype_parameters = known_dtype_params.get(str(file_parameters.data_type), {})
    if file_parameters.is_lightcone:
        lightcone_parameters = lightcone.LightconeParams
        required_dtype_params = dtype_parameters.get("required", {})
        required_dtype_params["lightcone"] = lightcone_parameters
        dtype_parameters["required"] = required_dtype_params
    return dtype_parameters


def get_dtype_column_plugins(
    header,
    producers,
    columns,
):
    plugins = __get_column_plugins(header)
    for name, producer in plugins.items():
        if name not in columns:
            continue
        producer = producer.bind(columns)
        producers.append(producer)
        columns[name] = producer.uuid

    return producers, columns


def __get_column_plugins(header):
    if header.file.data_type == "synthetic_galaxies":
        return {"top_host_idx": top_host_idx}
    return {}


def get_dtype_lightcone_plugins(header, columns):
    if header.file.data_type == "synthetic_galaxies" and "top_host_idx" in columns:
        return [offset_top_host_idx]
    return []
