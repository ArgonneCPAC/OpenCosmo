from pathlib import Path
from typing import cast

import h5py

import opencosmo as oc
from opencosmo.dataset import Dataset
from opencosmo.header import read_header

from .lightcone import Lightcone


def open_lightcone(files: list[Path], **load_kwargs):
    """
    Lightcone files are fairly restricted in how they can be structured
    for now at least. Hopefully we can relax some of this stuff in the future.

    - All files contain headers at the top level
    - Each file may only contain a single redshift step
    - All files should have the same data types

    Notes:
     - "single redshift step" does not necessarily mean
        those steps are the same as the simulation steps. Things may have been
        re-chunked
     - Some lightcone files may still have multiple datasets of the same type.
    """
    headers = list(map(read_header, files))
    if any(not h.file.is_lightcone for h in headers):
        raise ValueError("Not all files are lightcones")
    simulations = set(h.reformat.simulation_name for h in headers)
    dtypes = set(h.file.data_type for h in headers)
    steps = set(h.file.step for h in headers)
    if len(simulations) != 1 or len(dtypes) != 1:
        raise ValueError(
            "Lightcones must come from the same simulation and have the same data type"
        )
    if len(steps) != len(files):
        raise ValueError("Each file must contain only a single lightcone step!")

    datasets = {}
    for file in files:
        new_ds = oc.open(file)
        if not isinstance(new_ds, Lightcone):
            raise ValueError("Didn't find a lightcone in a lightcone file!")
        for key, ds in new_ds.items():
            key = "_".join([ds.dtype, str(ds.header.file.step)])
            datasets[key] = ds

    z_range = headers[0].lightcone.z_range
    return Lightcone(datasets, z_range)


def open_lightcone_file(path: Path) -> dict[str, Dataset]:
    handle = h5py.File(path)
    if "data" in handle.keys():
        ds = cast(Dataset, oc.open(handle))

        return {"_".join([ds.dtype, str(ds.header.file.step)]): ds}
    else:
        raise NotImplementedError
