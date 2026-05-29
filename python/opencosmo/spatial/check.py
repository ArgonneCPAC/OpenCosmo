from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import astropy.units as u  # type: ignore
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore

import opencosmo.dataset.state as st

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


def check_containment(
    state: DatasetState,
    region: Region,
    parameters: FileParameters,
    select_by: Optional[str] = None,
):
    dtype = str(parameters.data_type)
    if parameters.is_lightcone:
        return __check_containment_2d(state, region, dtype)
    else:
        return __check_containment_3d(state, region, dtype)


def get_theta_phi_coordinates(state: DatasetState):
    coord_values = st.get_data(st.select(state, {"theta", "phi"}), unpack=False)
    ra = coord_values["phi"]
    dec = np.pi / 2 - coord_values["theta"]

    return SkyCoord(ra, dec, unit=u.rad)


def get_theta_phi_coordinates_pixel(state: DatasetState):
    pixel_values = np.atleast_1d(st.get_metadata(state, ["pixel"])["pixel"])
    theta, phi = hp.pix2ang(
        state.header.healpix_map["nside"], pixel_values, lonlat=False, nest=True
    )
    ra = phi
    dec = np.pi / 2 - theta
    return SkyCoord(ra, dec, unit=u.rad)


def find_coordinates_2d(state: DatasetState):
    columns = set(state.columns)
    if state.header.file.data_type == "healpix_map":
        return get_theta_phi_coordinates_pixel(state)
    elif len(columns.intersection({"ra", "dec"})) == 2:
        data = st.get_data(st.select(state, {"ra", "dec"}), unpack=False)
        return SkyCoord(data["ra"], data["dec"])
    raise ValueError("Dataset does not contain coordinates")


def find_coordinates_3d(
    state: DatasetState, dtype: str, select_by: Optional[str] = None
):
    try:
        allowed_coordinates = ALLOWED_COORDINATES_3D[dtype]
    except KeyError:
        allowed_coordinates = ALLOWED_COORDINATES_3D["default"]
    if select_by is None:
        column_name_base = next(iter(allowed_coordinates.values()))
    else:
        column_name_base = allowed_coordinates[dtype]

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
    select_by: Optional[str] = None,
):
    columns = find_coordinates_3d(state, dtype, select_by)
    data = st.get_data(st.select(state, set(columns)))
    data = np.vstack(tuple(data[col].data for col in columns))
    return region.contains(data)


def __check_containment_2d(
    state: DatasetState,
    region: Region,
    dtype: str,
    select_by: Optional[str] = None,
):
    coords = find_coordinates_2d(state)
    return region.contains(coords)
