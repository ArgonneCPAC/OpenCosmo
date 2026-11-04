from __future__ import annotations

from typing import TYPE_CHECKING

from opencosmo.dtypes import hacc, lightcone

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
    if not file_parameters.is_lightcone:
        return dtype_parameters
    # Build new dicts; mutating the module-level registry leaks across headers
    return dtype_parameters | {
        "required": dtype_parameters.get("required", {})
        | {"lightcone": lightcone.LightconeParams}
    }
