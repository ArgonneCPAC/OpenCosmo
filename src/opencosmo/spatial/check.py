from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import astropy.units as u  # type: ignore
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore

if TYPE_CHECKING:
    from opencosmo.dataset.dataset import Dataset
    from opencosmo.parameters import FileParameters
    from opencosmo.spatial.protocols import Region

ALLOWED_COORDINATES_3D = {
    "default": {
        "fof": "fof_halo_center_",
        "mass": "fof_halo_com_",
        "sod": "sod_halo_com_",
    }
}


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


def get_theta_phi_coordinates(dataset: "Dataset"):
    coord_values = dataset.select(["theta", "phi"]).get_data(unpack=False)
    ra = coord_values["phi"]
    dec = np.pi / 2 - coord_values["theta"]

    return SkyCoord(ra, dec, unit=u.rad)


def get_theta_phi_coordinates_pixel(dataset: "Dataset"):
    pixel_values = np.atleast_1d(dataset.get_metadata(["pixel"])["pixel"])
    theta, phi = hp.pix2ang(
        dataset.header.healpix_map["nside"], pixel_values, lonlat=False, nest=True
    )
    ra = phi
    dec = np.pi / 2 - theta
    return SkyCoord(ra, dec, unit=u.rad)


def find_coordinates_2d(dataset: "Dataset"):
    columns = set(dataset.columns)
    if dataset.header.file.data_type == "healpix_map":
        return get_theta_phi_coordinates_pixel(dataset)
    elif len(columns.intersection(set(["theta", "phi"]))) == 2:
        return get_theta_phi_coordinates(dataset)
    elif len(columns.intersection(set(["ra", "dec"]))) == 2:
        data = dataset.select(["ra", "dec"]).get_data(unpack=False)
        return SkyCoord(data["ra"], data["dec"])
    raise ValueError("Dataset does not contain coordinates")


def find_coordinates_3d(ds: "Dataset", dtype: str, select_by: Optional[str] = None):
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
    return expected_cols


def __check_containment_3d(
    ds: "Dataset", region: "Region", dtype: str, select_by: Optional[str] = None
):
    columns = find_coordinates_3d(ds, dtype, select_by)
    ds = ds.select(columns)
    data = ds.data

    data = np.vstack(tuple(data[col].data for col in columns))
    return region.contains(data)


def __check_containment_2d(
    ds: "Dataset", region: "Region", dtype: str, select_by: Optional[str] = None
):
    coords = find_coordinates_2d(ds)
    return region.contains(coords)
