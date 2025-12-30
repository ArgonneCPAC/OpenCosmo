from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from opencosmo import collection
from opencosmo.header import read_header
from opencosmo.units.convention import UnitConvention

if TYPE_CHECKING:
    import h5py

    from opencosmo.header import OpenCosmoHeader


class FILE_TYPE(Enum):
    HALO_PROPERTIES = 0
    HALO_PARTICLES = 1
    GALAXY_PROPERTIES = 2
    GALAXY_PARTICLES = 3
    SOD_BINS = 4
    LIGHTCONE = 5
    STRUCTURE_COLLECTION = 6
    SIMULATION_COLLECTION = 7
    SYNTHETIC_CATALOG = 8
    HEALPIX_MAP = 9


def get_file_type(file: h5py.File) -> FILE_TYPE:
    if "header" in file.keys():
        dtype = file["header"]["file"].attrs["data_type"]
        if dtype == "halo_particles":
            return FILE_TYPE.HALO_PARTICLES
        elif dtype == "halo_profiles":
            return FILE_TYPE.SOD_BINS
        elif dtype == "halo_properties":
            return FILE_TYPE.HALO_PROPERTIES
        elif dtype == "galaxy_properties":
            return FILE_TYPE.GALAXY_PROPERTIES
        elif dtype == "galaxy_particles":
            return FILE_TYPE.GALAXY_PARTICLES
        elif dtype == "diffsky_fits":
            return FILE_TYPE.SYNTHETIC_CATALOG
        elif dtype == "healpix_map":
            return FILE_TYPE.HEALPIX_MAP
        else:
            raise ValueError(f"Unknown file type {dtype}")

    if not all("header" in group.keys() for group in file.values()):
        for subgroup in file.values():
            if not all("header" in g.keys() for g in subgroup.values()):
                raise ValueError(
                    "Unknown file type. "
                    "It appears to have multiple datasets, but organized incorrectly"
                )
    if all(group["header"]["file"].attrs["is_lightcone"] for group in file.values()):
        return FILE_TYPE.LIGHTCONE
    elif (
        len(set(group["header"]["file"].attrs["data_type"] for group in file.values()))
        == 1
    ):
        return FILE_TYPE.SIMULATION_COLLECTION

    elif all("data" not in group.keys() for group in file.values()):
        for group in file.values():
            sub_groups = {
                g["header"]["file"].attrs["data_type"]: g for g in group.values()
            }
            collection.structure.io.validate_linked_groups(sub_groups)
        return FILE_TYPE.SIMULATION_COLLECTION
    else:
        group = {name: group for name, group in file.items()}
        collection.structure.io.validate_linked_groups(group)
        return FILE_TYPE.STRUCTURE_COLLECTION


class OpenTarget:
    def __init__(self, group: h5py.Group | h5py.File, header: OpenCosmoHeader):
        self.group = group
        self.header = header

    @property
    def data_type(self):
        return self.header.file.data_type


def make_all_targets(files: list[h5py.File]):
    bad_files = []
    targets = []
    for file in files:
        try:
            targets += make_file_targets(file)
        except ValueError:
            bad_files.append(file.filename)
            raise
    if bad_files:
        raise ValueError(
            f"Some files were not able to be opened. They may not be OpenCosmo files: {bad_files}"
        )

    return targets


def make_file_targets(file: h5py.File):
    try:
        header = read_header(file, unit_convention=UnitConvention.COMOVING)
    except KeyError:
        header = None
    if header is not None and "data" in file.keys():
        return [OpenTarget(file, header)]
    if header is None and "data" in file.keys():
        raise ValueError(
            f"The file at {file.file.filename} appears to be missing a header. "
            "Are you sure it is an OpenCosmo file?"
        )
    if header is None:
        headers = {name: read_header(group) for name, group in file.items()}
    else:
        headers = {name: header for name in file.keys() if name != "header"}

    output = []
    for name, header in headers.items():
        target = OpenTarget(file[name], header)
        output.append(target)
    return output


def evaluate_load_conditions(targets: list[OpenTarget], open_kwargs: dict[str, bool]):
    """
    Datasets can define conditional loading via an addition group called "load/if".
    the "if" group can define parameters which must either be true or false for the
    given group to be loaded. These parameters can then be provided by the user to the
    "open" function. Parameters not specified by the user default to False.

    Note that some open kwargs may be used in other places in the opening process,
    and will just be ignored here.
    """
    if len(targets) == 1:
        return targets
    output = []
    for target in targets:
        try:
            ifgroup = target.group["load/if"]
        except KeyError:
            output.append(target)
            continue
        load = True
        for key, condition in ifgroup.attrs.items():
            load = load and (open_kwargs.get(key, False) == condition)
        if load:
            output.append(target)
    return output
