from typing import TYPE_CHECKING, Optional

import numpy as np

ALLOWED_COORDINATES = {
    "halo_properties": {
        "fof": "fof_halo_center_",
        "mass": "fof_halo_com_",
        "sod": "sod_halo_com_",
    }
}


if TYPE_CHECKING:
    from opencosmo.dataset.dataset import Dataset
    from opencosmo.spatial.region import BoxRegion


def check_containment(
    ds: "Dataset", region: "BoxRegion", dtype: str, select_by: Optional[str] = None
):
    allowed_coordinates = ALLOWED_COORDINATES[dtype]
    if select_by is None:
        column_name_base = next(iter(allowed_coordinates.values()))
    else:
        column_name_base = allowed_coordinates[dtype]

    cols = set(filter(lambda colname: colname.startswith(column_name_base), ds.columns))
    expected_cols = set(column_name_base + dim for dim in ["x", "y", "z"])
    if cols != expected_cols:
        raise ValueError(
            "Unable to find the correct coordinate columns in this dataset! "
            f"Found {cols} but expected {expected_cols}"
        )

    ds = ds.select([column_name_base + dim for dim in ["x", "y", "z"]])

    data = np.vstack(tuple(col.data for col in ds.data.itercols()))
    return region.contains(data)
