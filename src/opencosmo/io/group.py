from enum import Enum
from typing import TypedDict

import h5py

from opencosmo import collection as occ
from opencosmo.header import OpenCosmoHeader, read_header

"""
There are a few file structures we have to be able to support.

1. header + data groups -> single dataset
2. header + several non-data groups structure collection or lightcone collection. If lightcone collection,
   all dataset will have the same data type and will have is_lightcone set to true in the header.
3. no header, serveral groups -> Structure Collection


When a user passes multiple files, there are basically two options (at present), structure collection
or lightcone collection.

The former will consist of halo properties or galaxy properties, particles, and/or profiles
The later will consist of several datasets, each with the same data type and is_lightcone set to true
"""


class DatasetTarget(TypedDict):
    header: OpenCosmoHeader
    dataset_group: h5py.Group


class FileType(Enum):
    DATASET = "dataset"
    LIGHTCONE = "lightcone"
    STRUCTURE_COLLECTION = "structure_collection"
    PARTICLES = "particles"
    SIMULATION_COLLECTION = "simulation_collection"


class CollectionType(Enum):
    pass


class FileTarget(TypedDict):
    file_type: FileType
    dataset_targets: list[DatasetTarget]


def make_file_targets(files: list[h5py.File]):
    targets = []
    for file in files:
        targets.append(__make_file_target(file))

    if len(targets) > 1:
        collection_type = __determine_multi_file_collection_type(targets)
        return collection_type.open(targets)
    return __determine_single_file_collection_type(targets[0])


def __open_single_file(target: FileTarget):
    match target["file_type"]:
        case FileType.SIMULATION_COLLECTION:
            return occ.SimulationCollection


def __determine_single_file_collection_type(target: FileTarget):
    print(target)
    assert False


def __determine_multi_file_collection_type(targets: list[FileTarget]):
    properties = []
    particles_or_profiles = []
    lightcones = []
    other_datasets = []
    for target in targets:
        if target["file_type"] in [
            FileType.STRUCTURE_COLLECTION,
            FileType.SIMULATION_COLLECTION,
        ]:
            raise ValueError("Invalid combination of files!")
        if (
            target["file_type"] == FileType.DATASET
            and target["dataset_targets"][0]["header"].file.data_type == "halo_profiles"
        ):
            particles_or_profiles.append(target)
        elif target["file_type"] == FileType.DATASET and target["dataset_targets"][0][
            "header"
        ].file.data_type in ["halo_properties", "galaxy_properties"]:
            properties.append(target)
        elif target["file_type"] == FileType.PARTICLES:
            particles_or_profiles.append(target)
        elif target["file_type"] == FileType.LIGHTCONE:
            lightcones.append(target)
        elif target["file_type"] == FileType.DATASET:
            other_datasets.append(target)
        else:
            raise ValueError("Invalid combination of files!")

    return __get_collection_type_from_categorized_lists(
        properties, particles_or_profiles, lightcones, other_datasets
    )


def __get_collection_type_from_categorized_lists(
    properties: list[FileTarget],
    particles_or_profiles: list[FileTarget],
    lightcones: list[FileTarget],
    other_datasets: list[FileTarget],
):
    flags = (
        len(properties) > 0,
        len(particles_or_profiles) > 0,
        len(lightcones) > 0,
        len(other_datasets) > 0,
    )
    match flags:
        case (True, True, False, False):
            return occ.StructureCollection
        case (False, False, True, False):
            return occ.Lightcone
        case (True, False, False, False):
            return __get_multi_dataset_type(properties)
        case (False, False, False, True):
            return __get_multi_dataset_type(other_datasets)
        case _:
            raise ValueError("Invalid combination of files")


def __get_multi_dataset_type(file_targets: list[FileTarget]):
    dtypes = set(
        ft["dataset_targets"][0]["header"].file.data_type for ft in file_targets
    )
    is_lightcone = set(
        ft["dataset_targets"][0]["header"].file.is_lightcone for ft in file_targets
    )
    if dtypes == {"halo_properties", "galaxy_properties"}:  # special case
        return occ.StructureCollection

    if len(dtypes) == 1 or len(is_lightcone) > 1:
        raise ValueError(
            "When opening multiple files, they must either be several different data types from a single simulation, "
            "a single data type from several simulations, or a single lightcone data type from a single simulation"
        )
    if is_lightcone.pop():
        return occ.Lightcone
    else:
        return occ.SimulationCollection


def __make_file_target(file: h5py.File):
    dataset_targets = __find_all_datasets(file)
    file_type = __identify_file_type(dataset_targets)
    return FileTarget(file_type=file_type, dataset_targets=dataset_targets)


def __identify_file_type(targets: list[DatasetTarget]):
    data_types = set(t["header"].file.data_type for t in targets)
    is_lightcone = [t["header"].file.is_lightcone for t in targets]
    if all("particle" in dt for dt in data_types):
        return FileType.PARTICLES
    if len(data_types) == 1 and all(is_lightcone):
        return FileType.LIGHTCONE
    if len(targets) == 1:
        return FileType.DATASET

    parents = set(t["dataset_group"].parent.name for t in targets)
    if len(parents) == 1 and len(data_types) == len(targets):
        return FileType.STRUCTURE_COLLECTION
    return FileType.SIMULATION_COLLECTION


def __find_all_datasets(file: h5py.File) -> list[DatasetTarget]:
    """
    Search through a file and locate all the datasets. Each dataset is identified
    with a "data" group. The header associated with the file is the closest
    header at or above.
    """

    known_headers = []
    if "header" in file.keys():
        known_headers = ["/"]
    file.visit(
        lambda n: known_headers.append(n)
        if isinstance(file[n], h5py.Group) and "header" in file[n].keys()
        else None
    )
    if not known_headers:
        raise ValueError(
            f"Cannot find a header in {file.name}. Are you sure it is an OpenCosmo file?"
        )
    known_datasets = []
    for header_container in known_headers:
        header = read_header(file[header_container])
        if "data" in file[header_container].keys():
            known_datasets.append(
                DatasetTarget(header=header, dataset_group=file[header_container])
            )
        file[header_container].visititems(
            lambda _, object: known_datasets.append(
                DatasetTarget(header=header, dataset_group=object)
            )
            if isinstance(object, h5py.Group) and "data" in object.keys()
            else None
        )
    if not known_datasets:
        raise ValueError(
            f"File {file.name} contains an OpenCosmo header, but does not seem to be formatted correctly!"
        )
    return known_datasets
