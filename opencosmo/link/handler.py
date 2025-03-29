from __future__ import annotations
from typing import Protocol
from h5py import File, Group
import opencosmo as oc
from typing import Iterable, Optional
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.transformations import units as u
from opencosmo.spatial import read_tree
from opencosmo.handler import OutOfMemoryHandler
from pathlib import Path
from collections import defaultdict

import numpy as np
LINK_ALIASES = { # Left: Name in file, right: Name in collection
    "sodbighaloparticles_star_particles": "star_particles",
    "sodbighaloparticles_dm_particles": "dm_particles",
    "sodbighaloparticles_gravity_particles": "dm_particles",
    "sodbighaloparticles_agn_particles": "agn_particles",
    "sodbighaloparticles_gas_particles": "gas_particles",
    "sod_profile": "halo_profiles",
    "galaxyproperties": "galaxy_properties",
}

ALLOWED_LINKS = {  # Files that can serve as a link holder and
    "halo_properties": ["halo_particles", "halo_profiles"],
    "galaxy_properties": ["galaxy_particles"],
}

def verify_links(*headers: OpenCosmoHeader) -> tuple[str, list[str]]:
    """
    Verify that the links in the headers are valid. This means that the
    link holder has a corresponding link target and that the link target
    is of the correct type. It also verifies that the linked files are from
    the same simulation. Returns a dictionary where the keys are the
    link holder files and the values are lists of the corresponding link.

    Raises an error if the links are not valid, otherwise returns the links.
    """

    data_types = [header.file.data_type for header in headers]
    if len(set(data_types)) != len(data_types):
        raise ValueError("Data types in files must be unique to link correctly")

    master_files = [dt for dt in data_types if dt in ALLOWED_LINKS]
    if not master_files:
        raise ValueError("No valid link holder files found in headers")

    dtypes_to_headers = {header.file.data_type: header for header in headers}

    links = defaultdict(list)  # {file: [link_header, ...]}
    for file in master_files:
        for link in ALLOWED_LINKS[file]:
            try:
                link_header = dtypes_to_headers[link]
                # Check that the headers come from the same simulation
                if link_header.simulation != dtypes_to_headers[file].simulation:
                    raise ValueError(f"Simulation mismatch between {file} and {link}")
                links[file].append(link)
            except KeyError:
                continue  # No link header found for this file

    # Master files also need to have the same simulation
    if len(master_files) > 1:
        raise ValueError("Data linking can only have one master file")
    for file in master_files:
        if (
            dtypes_to_headers[file].simulation
            != dtypes_to_headers[master_files[0]].simulation
        ):
            raise ValueError(
                f"Simulation mismatch between {file} and {master_files[0]}"
            )
    output_file = master_files[0]
    return output_file, links[output_file]

def open_linked(*files: Path):
    """
    Open a collection of files that are linked together, such as a
    properties file and a particle file.
    """
    file_handles = [File(file, "r") for file in files]
    headers = [read_header(file) for file in file_handles]
    properties_file, linked_files = verify_links(*headers)
    properties_index = next(index for index, header in enumerate(headers) if header.file.data_type == properties_file)
    properties_file = file_handles.pop(properties_index)
    properties_dataset = oc.open(properties_file)

    linked_files_by_type = {file["header"]["file"].attrs["data_type"]: file for file in file_handles}
    if len(linked_files_by_type) != len(linked_files):
        raise ValueError("Linked files must have unique data types")

    datasets = {}
    for dtype, pointer in linked_files_by_type.items():
        if "data" not in pointer.keys():
            datasets.update({k: pointer[k] for k in pointer.keys() if k != "header"})
        else:
            datasets.update({dtype: pointer})



    link_handlers = get_link_handlers(properties_file, datasets, properties_dataset.header)
    output = {}
    for key, handler in link_handlers.items():
        if key in LINK_ALIASES:
            output[LINK_ALIASES[key]] = handler
        else:
            output[key] = handler

    return LinkedCollection(properties_dataset, output)  




class LinkedCollection:
    """
    A collection of datasets that are linked together, allowing
    for cross-matching and other operations to be performed.

    For now, these are always a combination of a properties dataset
    and several particle or profile datasets. 
    """

    def __init__(
        self,
        properties: oc.Dataset,
        handlers: dict[str, LinkHandler],
        *args,
        **kwargs,
    ):
        """
        Initialize a linked collection with the provided datasets and links.
        """

        self.__properties = properties
        self.__handlers = handlers
        self.__idxs = self.__properties.indices

    def as_dict(self) -> dict[str, oc.Dataset]:
        return self

    def keys(self) -> list[str]:
        """
        Return the keys of the linked datasets.
        """
        return list(self.__handlers.keys()) + [self.__properties.header.file.data_type]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for dataset in self.values():
            try:
                dataset.__exit__(*args)
            except AttributeError:
                continue

    def filter(self, *masks):
        """
        Apply a filter to the properties dataset and propagate it to the linked datasets.
        """
        if not masks:
            return self
        filtered = self.__properties.filter(*masks)
        return LinkedCollection(
            filtered,
            self.__handlers,
        )
    def take(self, n: int, at: str = "start"):
        new_properties = self.__properties.take(n, at)
        return LinkedCollection(
            new_properties,
            self.__handlers,
        )


    def objects(self, data_types: Optional[Iterable[str]] = None) -> Iterable[tuple[oc.Dataset, dict[str, oc.Dataset]]]:
        """
        Iterate over the properties dataset and the linked datasets.
        """
        if data_types is None:
            handlers = self.__handlers
        elif not all(dt in self.__handlers for dt in data_types):
            raise ValueError("Some data types are not linked in the collection.")
        else:
            handlers = {dt: self.__handlers[dt] for dt in data_types}

        for i, row in enumerate(self.__properties.rows()):
            index = np.array([i])
            output = {key: handler.get_data(index) for key, handler in handlers.items()}
            if not any(output.values()):
                continue
            yield row, output

    def write(self, file: File):
        header = self.__properties.header
        header.write(file)
        self.__properties.write(file, header.file.data_type)
        link_group = file[header.file.data_type].create_group("data_linked")
        for key, handler in self.__handlers.items():
            handler.write(file, key, link_group, self.__idxs)

        
def get_link_handlers(link_file: File | Group, linked_files: dict[str, File | Group], header: OpenCosmoHeader) -> dict[str, "LinkHandler"]:
    if "data_linked" not in link_file.keys():
        raise KeyError("No linked datasets found in the file.")
    links = link_file["data_linked"]
    unique_dtypes = {key.rsplit("_", 1)[0] for key in links.keys()}
    output_links = {}
    for dtype in unique_dtypes:
        if dtype not in linked_files and LINK_ALIASES.get(dtype) not in linked_files:
            continue  # Skip if the linked file is not provided

        key = LINK_ALIASES.get(dtype, dtype)
        if "data" not in linked_files[key].keys():
            raise KeyError(f"No data group found in linked file for dtype '{dtype}'")
        try:
            start = links[f"{dtype}_start"]
            size = links[f"{dtype}_size"]
            output_links[key] = OomLinkHandler(linked_files[key], (start, size), header)
        except KeyError:
            index = links["sod_profile_idx"]
            output_links[key] = OomLinkHandler(linked_files[key], index, header)
    return output_links

def build_dataset(file: File | Group, indices: np.ndarray, header: OpenCosmoHeader) -> Dataset:
    tree = read_tree(file, header)
    builders, base_unit_transformations = u.get_default_unit_transformations(file, header)
    handler = OutOfMemoryHandler(file, tree=tree)
    return oc.Dataset(handler, header, builders, base_unit_transformations, indices)




class LinkHandler(Protocol):
    """
    A LinkHandler is responsible for handling linked datasets. Links are found
    in property files, and contain indexes into another dataset. The link handler
    is responsible for holding pointers to the linked files, and returning
    the associated data when requested (as a Dataset object).
    """

    def __init__(self, file: File | Group, links: Group | tuple[Group, Group], header: OpenCosmoHeader): ...
    def get_data(self, indices: int | np.ndarray) -> oc.Dataset: ...
    def write(self, data_group: Group, link_group: Group, indices: int | np.ndarray) -> None: ...


class OomLinkHandler:
    def __init__(self, file: File | Group, link: Group | tuple[Group, Group], header: OpenCosmoHeader = None):
        self.file = file
        self.link = link
        self.header = header

    def get_data(self, indices: np.ndarray) -> Optional[oc.Dataset]:
        if isinstance(self.link, tuple):
            start = self.link[0][indices]
            size = self.link[1][indices]
            valid_rows = size > 0
            start = start[valid_rows]
            size = size[valid_rows]
            if not start.size:
                return None
            indices = np.concatenate([np.arange(idx, idx + length) for idx, length in zip(start, size)])
        else:
            indices = self.link[indices]
            indices = indices[indices >= 0]
            if not indices.size:
                return None

        return build_dataset(self.file, indices, self.header)

    def write(self, file: File, name: str, link_group: Group, indices: np.ndarray):
        # Pack the indices
        if not isinstance(self.link, tuple): 
            new_idxs = np.arange(len(indices))
            link_group.create_dataset("sod_profile_idx", data=new_idxs)
        else:
            lengths = self.link[1][indices]
            new_starts = np.insert(np.cumsum(lengths), 0, 0)[:-1]
            link_group.create_dataset(f"{name}_start", data=new_starts)
            link_group.create_dataset(f"{name}_size", data=lengths)

        dataset = self.get_data(indices)
        dataset.write(file, name)
