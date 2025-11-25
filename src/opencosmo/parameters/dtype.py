from __future__ import annotations

from typing import TYPE_CHECKING

from opencosmo.parameters import hacc

if TYPE_CHECKING:
    from pydantic import BaseModel

    from opencosmo.parameters.file import FileParameters


# TODO: think I need to alter this
def get_dtype_parameters(file_parameters: FileParameters) -> dict[str, type[BaseModel]]:
    if file_parameters.origin == "HACC":
        known_dtype_params = hacc.DATATYPE_PARAMETERS
    else:
        raise ValueError(f"Unknown dataset origin {file_parameters.origin}")
    print(known_dtype_params)
    dtype_parameters = known_dtype_params[str(file_parameters.data_type)]
    print(dtype_parameters)
    if file_parameters.is_lightcone:
        lightcone_parameters = hacc.LightconeParams
        dtype_parameters.update({"lightcone": lightcone_parameters})
    return dtype_parameters
