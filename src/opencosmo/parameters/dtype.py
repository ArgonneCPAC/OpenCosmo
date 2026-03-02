from __future__ import annotations

from typing import TYPE_CHECKING

from opencosmo.parameters import hacc, lightcone

if TYPE_CHECKING:
    from pydantic import BaseModel

    from opencosmo.parameters.file import FileParameters


# TODO: think I need to alter this
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
