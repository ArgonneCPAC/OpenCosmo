from pathlib import Path
from typing import Type

import opencosmo as oc
from opencosmo.collection.protocols import Collection
from opencosmo.collection.simulation import SimulationCollection
from opencosmo.collection.structure.io import validate_linked_groups
from opencosmo.io.io import FILE_TYPE, OpenTarget


def open_simulation_files(**paths: Path) -> SimulationCollection:
    """
    Open multiple files and return a simulation collection. The data
    type of every file must be the same.

    Parameters
    ----------
    paths : str or Path
        The paths to the files to open.

    Returns
    -------
    SimulationCollection

    """
    datasets: dict[str, oc.Dataset] = {}
    for key, path in paths.items():
        dataset = oc.open(path)
        if not isinstance(dataset, oc.Dataset):
            raise ValueError("All datasets must be of the same type.")
    dtypes = set(dataset for dataset in datasets.values())
    if len(dtypes) != 1:
        raise ValueError("All datasets must be of the same type.")
    return SimulationCollection(datasets)


def get_collection_type(
    targets: list[OpenTarget], file_types: list[FILE_TYPE]
) -> Type[Collection]:
    """
    If there are multiple files, determine their collection type. There are
    three options we support at present:

    1. All files contain a single lightcone dataset, all of the same type
    2. The files contain a single non-lightcone datatype.
    3. The files are linked together into a structure collection
    """
    handles_by_type = {target.data_type: target.group for target in targets}
    is_lightcone = [target.header.file.is_lightcone for target in targets]
    unique_data_types = set(handles_by_type.keys())
    unique_file_types = set(file_types)
    if len(unique_data_types) == 1 and all(is_lightcone):
        return oc.Lightcone
    elif len(unique_data_types) == 1 and all(not il for il in is_lightcone):
        return oc.SimulationCollection
    elif unique_file_types == set([FILE_TYPE.PARTICLES, FILE_TYPE.DATASET]):
        validate_linked_groups(handles_by_type)
        return oc.StructureCollection
    elif (
        len(unique_file_types) == 1
        and unique_file_types.pop() == FILE_TYPE.STRUCTURE_COLLECTION
    ):
        validate_linked_groups(handles_by_type)
        return oc.StructureCollection
    else:
        raise ValueError("Invalid combination of files")
