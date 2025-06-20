from typing import TYPE_CHECKING, Optional

import astropy.units as u  # type: ignore
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore

from opencosmo.parameters import FileParameters

ALLOWED_COORDINATES_3D = {
    "default": {
        "fof": "fof_halo_center_",
        "mass": "fof_halo_com_",
        "sod": "sod_halo_com_",
    }
}

ALLOWED_COORDINATES_2D = {"default": set(["theta", "phi"])}


if TYPE_CHECKING:
    from opencosmo.dataset.dataset import Dataset
    from opencosmo.spatial.protocols import Region


def check_containment(
    ds: "Dataset",
    region: "Region",
    parameters: FileParameters,
    select_by: Optional[str] = None,
):
    dtype = str(parameters.data_type)
    if parameters.is_lightcone:
        return __check_containment_2d(ds, region, dtype)
    else:
        return __check_containment_3d(ds, region, dtype)


def __check_containment_3d(
    ds: "Dataset", region: "Region", dtype: str, select_by: Optional[str] = None
):
    try:
        allowed_coordinates = ALLOWED_COORDINATES_3D[dtype]
    except KeyError:
        allowed_coordinates = ALLOWED_COORDINATES_3D["default"]
    if select_by is None:
        column_name_base = next(iter(allowed_coordinates.values()))
    else:
        column_name_base = allowed_coordinates[dtype]

    cols = set(filter(lambda colname: colname.startswith(column_name_base), ds.columns))
    expected_cols = [column_name_base + dim for dim in ["x", "y", "z"]]
    if cols != set(expected_cols):
        raise ValueError(
            "Unable to find the correct coordinate columns in this dataset! "
            f"Found {cols} but expected {expected_cols}"
        )

    ds = ds.select(expected_cols)
    data = ds.data

    data = np.vstack(tuple(data[col].data for col in expected_cols))
    return region.contains(data)


def __check_containment_2d(
    ds: "Dataset", region: "Region", dtype: str, select_by: Optional[str] = None
):
    try:
        allowed_coordinates = ALLOWED_COORDINATES_2D[dtype]
    except KeyError:
        allowed_coordinates = ALLOWED_COORDINATES_2D["default"]
    cols = set(ds.columns)
    if cols.intersection(allowed_coordinates) != allowed_coordinates:
        raise ValueError(
            "Unable to find the correct coordinate columns in this dataset!"
        )

    coord_values = ds.select(allowed_coordinates).data
    ra = coord_values["phi"]
    dec = np.pi / 2 - coord_values["theta"]

    coords = SkyCoord(ra, dec, unit=u.rad)
    return region.contains(coords)
