from __future__ import annotations

from functools import wraps
from inspect import Parameter, signature
from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore

if TYPE_CHECKING:
    from opencosmo.dataset.state import DatasetState
    from opencosmo.dtypes import FileParameters
    from opencosmo.spatial.protocols import Region

ALLOWED_COORDINATES_3D = {
    "default": {
        "fof": "fof_halo_center_",
        "mass": "fof_halo_com_",
        "sod": "sod_halo_com_",
    }
}


def wrap_evaluate_with_coordinates(coordinate_names: list[str]):
    def outer_wrapper(func):
        current_signature = signature(func)
        names = current_signature.parameters.keys()
        if "data" not in names:
            coordinates_to_add = set(coordinate_names).difference(names)
            new_parameters = list(current_signature.parameters.values()) + [
                Parameter(name, kind=Parameter.KEYWORD_ONLY)
                for name in coordinates_to_add
            ]

            new_signature = current_signature.replace(parameters=new_parameters)
        else:
            coordinates_to_add = set()
            new_signature = current_signature

        @wraps(func)
        def wrapper(*args, **kwargs):
            _ = new_signature.bind(*args, **kwargs)
            coordinates = {
                name: kwargs.pop(name) if name in coordinates_to_add else kwargs[name]
                for name in coordinate_names
            }

            results = func(*args, **kwargs)
            if not isinstance(results, dict):
                results = {func.__name__: results}
            if not coordinates_to_add:
                return results
            return __build_output(results, coordinates)

        wrapper.__signature__ = new_signature
        return wrapper

    return outer_wrapper


def __build_output(results: dict, coordinates: dict):
    """
    Two allowed situations:
    1. The results are arrays and the coordinates are arrays of the same length -> just combine
    2. The results are arrays and the coordinates are single elements ->
    """
    first_output = next(iter(results.values()))
    coordinate_length = len(
        np.atleast_1d(next(iter(coordinates.values())))
    )  # weird size mismatches are caught elsehwere
    has_units = hasattr(next(iter(coordinates.values())), "unit")

    try:
        length = len(first_output)
        if coordinate_length == length:
            return results | coordinates
        elif coordinate_length == 1:
            new_coordinates = {
                name: np.full(length, c) for name, c in coordinates.items()
            }
            if has_units:
                new_coordinates = {
                    name: c * coordinates[name].unit
                    for name, c in new_coordinates.items()
                }
            return results | new_coordinates

        else:
            raise ValueError("Placeholder")

    except TypeError:
        if coordinate_length == 1:
            return results | coordinates
        else:
            raise ValueError


def check_containment(
    state: DatasetState,
    region: Region,
    parameters: FileParameters,
    select_by: str | None = None,
):
    dtype = str(parameters.data_type)
    if parameters.is_lightcone:
        return __check_containment_2d(state, region, dtype, select_by)
    else:
        return __check_containment_3d(state, region, dtype, select_by)


def find_coordinates_2d(state: DatasetState):
    from opencosmo.dataset import operations as dsops

    columns = set(state.columns)
    if len(columns.intersection(set(["ra", "dec"]))) == 2:
        selected = dsops.select(state, ["ra", "dec"])
        data = dsops.get_data(selected, "astropy", unpack=False)
        return SkyCoord(data["ra"], data["dec"])
    raise ValueError("Dataset does not contain coordinates")


def find_coordinates_3d(state: DatasetState, dtype: str, select_by: str | None = None):
    try:
        allowed_coordinates = ALLOWED_COORDINATES_3D[dtype]
    except KeyError:
        allowed_coordinates = ALLOWED_COORDINATES_3D["default"]
    if select_by is None:
        column_name_base = next(iter(allowed_coordinates.values()))
    else:
        column_name_base = allowed_coordinates[select_by]

    cols = set(
        filter(lambda colname: colname.startswith(column_name_base), state.columns)
    )
    expected_cols = [column_name_base + dim for dim in ["x", "y", "z"]]
    if cols != set(expected_cols):
        raise ValueError(
            "Unable to find the correct coordinate columns in this dataset! "
            f"Found {cols} but expected {expected_cols}"
        )
    return expected_cols


def __check_containment_3d(
    state: DatasetState,
    region: Region,
    dtype: str,
    select_by: str | None = None,
):
    from opencosmo.dataset import operations as dsops

    columns = find_coordinates_3d(state, dtype, select_by)
    selected = dsops.select(state, columns)
    data = dsops.get_data(selected, "astropy")

    data = np.vstack(tuple(data[col].data for col in columns))
    return region.contains(data)


def __check_containment_2d(
    state: DatasetState,
    region: Region,
    dtype: str,
    select_by: str | None = None,
):
    coords = find_coordinates_2d(state)
    return region.contains(coords)
