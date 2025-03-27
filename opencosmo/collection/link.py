"""
Some types of data contain links to other data. In particular, property files
contain links to their particle files. HaloProperty files contain links to
halo particles, which include AGN, Dark Matter, Gas, and Star particles.
GalaxyProperties contain a link to the associated star particles in the
GalaxyParticles file. A link is simply a combination of a starting index
and a length, which maps a single row in a property file to a range of rows
in a particle file.

As such, efficient querying of linking can only be done between specific types
of data
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, TypedDict

import numpy as np
from h5py import File, Group

import opencosmo as oc
from opencosmo.dataset.mask import Mask
from opencosmo.header import OpenCosmoHeader, read_header
import h5py

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


class LinkedCollection(dict):
    """
    A collection of datasets that are linked together, allowing
    for cross-matching and other operations to be performed.

    For now, these are always a combination of a properties dataset
    and several particle or profile datasets. 
    """

    def __init__(
        self,
        header: OpenCosmoHeader,
        properties: oc.Dataset,
        datasets: dict,
        links: GalaxyPropertyLink | HaloPropertyLink,
        *args,
        **kwargs,
    ):
        """
        Initialize a linked collection with the provided datasets and links.
        """

        self.__header = header
        self.__properties = properties
        self.__datasets = datasets
        self[properties.header.file.data_type] = properties
        self.__linked = links
        self.__idxs = self.__properties.indices
        self.__aliases = {}
        for link_name in links:
            if link_name in self.__datasets:
                self.__aliases[link_name] = link_name
            elif LINK_ALIASES.get(link_name, "") in self.__datasets:
                self.__aliases[LINK_ALIASES[link_name]] = link_name
        if len(self.__aliases) != len(self.__datasets):
            raise ValueError(
                "Not all linked datasets have a corresponding link in the properties file"
            )

        self.update(self.__datasets)

    def as_dict(self) -> dict[str, oc.Dataset]:
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for dataset in self.values():
            try:
                dataset.__exit__(*args)
            except AttributeError:
                continue

    @property
    def header(self):
        return self.__header

    @classmethod
    def read(cls, file: File, names: Optional[Iterable[str]] = None):
        """
        Read a collection of linked datasets from an HDF5 file.
        """
        raise NotImplementedError(
            "Directly reading linked collections is not supported. Use `open` instead."
        )
        header = read_header(file)
        properties = oc.read(file, header.file.data_type)
        links = get_links(file[header.file.data_type])
        if names is None:
            names = set(file.keys()) - {header.file.data_type, "header"}

        datasets = {name: oc.read(file, name) for name in names}
        output_datasets = {}
        for name, ds in datasets.items():
            try:
                output_datasets[LINK_ALIASES[name]] = ds
            except KeyError:
                output_datasets[name] = ds

        return cls(header, properties, output_datasets, links)

    @classmethod
    def open(cls, file: File, names: Optional[Iterable[str]] = None):
        """
        Open a collection of linked datasets from in a single HDF5 file.
        """
        header = read_header(file)
        properties = oc.open(file, header.file.data_type)
        if not isinstance(properties, oc.Dataset):
            raise ValueError(
                "Expected a single dataset for the properties file, but found a collection of them"
            )
        links = get_links(file[header.file.data_type])
        if names is None:
            names = set(file.keys()) - {header.file.data_type, "header"}

        datasets = {name: oc.open(file, name) for name in names}
        output_datasets = {}
        for name, ds in datasets.items():
            try:
                output_datasets[LINK_ALIASES[name]] = ds
            except KeyError:
                output_datasets[name] = ds
        return cls(header, properties, output_datasets, links)

    def write(self, file: File):
        """
        Write the collection to an HDF5 file.
        """
        self.__header.write(file)
        idxs = self.__properties.indices
        for key, dataset in self.items():
            if dataset is self.__properties:
                continue
            alias = self.__aliases[key]
            try:
                starts = self.__linked[alias]["start_index"][idxs]  # type: ignore
                sizes = self.__linked[alias]["length"][idxs]  # type: ignore
                indices = np.concatenate(
                    [
                        np.arange(start, start + size)
                        for start, size in zip(starts, sizes)
                    ]
                )
                dataset.write(file, alias, _indices=indices)
            except (IndexError, ValueError):
                indices = self.__linked[alias][idxs]  # type: ignore
                dataset.write(file, alias, _indices=indices)

        property_dataset = self.__properties.header.file.data_type
        self.__properties.write(file, property_dataset, property_dataset)
        write_links(file[property_dataset], self.__linked, self.__properties.indices)

    def __get_linked(self, dtype: str, index: int):
        if dtype not in self.__aliases:
            raise ValueError(f"No links found for {dtype}")
        elif index >= len(self.__properties):
            raise ValueError(f"Index {index} out of range for {dtype}")
        # find the index into the linked dataset at the mask index
        alias = self.__aliases[dtype]
        linked_index = self.__idxs[index]
        try:
            start = self.__linked[alias]["start_index"][linked_index]  # type: ignore
            size = self.__linked[alias]["length"][linked_index]  # type: ignore
        except IndexError:
            start = self.__linked[alias][linked_index]  # type: ignore
            size = 1

        if start == -1 or size == -1:
            return None

        return self.__datasets[dtype].take_range(start, start + size)

    def objects(self, dtypes: Optional[str | list[str]] = None):
        """
        Iterate over the objects in the collection, returning the properties
        of the object as well as its particles and/or profiles. The specific
        datatypes you want to return can be specified with the `dtypes` argument.
        If `dtypes` is not provided, all linked datasets will be returned.

        The objects are returned as a tuple of the properties and a dictionary
        of the linked datasets. If only one datatype is requested, the second
        element of the tuple will simply be the linked dataset. For example
        If we have a collection of halo properties linked to halo particles and
        star particles:

        .. code-block:: python
            
            for properties, particles in collection.objects():
                properties # dict of properties for the given halo
                particles # dict containing one halo particle dataset 
                          # and one star particle dataset for this halo

        Parameters
        ----------
        dtypes : str or list of str, optional
            The data types to return. If not provided, all datasets will be returned.

        Returns
        -------

        tuple of (OpenCosmoHeader, dict) or (OpenCosmoHeader, oc.Dataset)
            The properties of the object and the linked datasets

        """
        if dtypes is None:
            dtypes = list(k for k in self.__linked.keys() if k in self.__datasets)
        elif isinstance(dtypes, str):
            dtypes = [dtypes]


        ndtypes = len(dtypes)
        for i, properties in enumerate(self.__properties.rows()):
            results = {dtype: self.__get_linked(dtype, i) for dtype in dtypes}
            if all(result is None for result in results.values()):
                continue
            if ndtypes == 1:
                yield properties, results[dtypes[0]]
            else:
                yield properties, results

    def filter(self, *masks: Mask):
        """
        Filtering a linked collection always operates on the properties dataset. For
        example, a collection of halos can be filtered by the standard halo properties,
        such as fof_halo_mass. The filteriing works identically to 
        :meth:`oc.Dataset.filter`.

        Parameters
        ----------
        masks : Mask
            The masks to apply to the properties dataset.

        Returns
        -------
        LinkedCollection
            A new collection with the filtered properties dataset.
        """
        new_properties = self.__properties.filter(*masks)
        return LinkedCollection(
            self.header, new_properties, self.__datasets, self.__linked
        )

    def take(self, n: int, at: str = "start"):
        """
        Take some number of objects from the collection. This method operates
        identically to :meth:`oc.Dataset.take`.
        """
        new_properties = self.__properties.take(n, at)
        return LinkedCollection(
            self.header, new_properties, self.__datasets, self.__linked
        )

    def with_units(self, convention: str) -> LinkedCollection:
        """
        Convert the units of the collection to a given convention. This method
        operates identically to :meth:`oc.Dataset.with_units
        """
        new_properties = self.__properties.with_units(convention)
        new_datasets = {k: v.with_units(convention) for k, v in self.__datasets.items()}
        return LinkedCollection(
            self.header, new_properties, new_datasets, self.__linked
        )


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


def get_links(file: File | Group) -> GalaxyPropertyLink | HaloPropertyLink:
    if "data_linked" not in file.keys():
        raise ValueError(f"No links found in {file.name}")
    keys = file["data_linked"].keys()
    # Remove everything after the last underscore to get unique link names
    unique_keys = {key.rsplit("_", 1)[0] for key in keys}
    output_links = {}
    for key in unique_keys:
        if f"{key}_start" in keys and f"{key}_size" in keys:
            output_links[key] = {
                "start_index": file["data_linked"][f"{key}_start"],
                "length": file["data_linked"][f"{key}_size"],
            }
        elif f"{key}_idx" in keys:
            output_links[key] = file["data_linked"][f"{key}_idx"]
        else:
            raise ValueError(f"Invalid link type found for {key}")
    return output_links

def pack_links(
    data_link: DataLink | h5py.dataset, indices: np.ndarray
) -> DataLink | np.ndarray:
    if isinstance(data_link, dict):
        lengths = data_link["length"][indices]
        new_starts = np.cumsum(lengths)
        new_starts = np.insert(new_starts, 0, 0)[:-1]
        return {"start_index": new_starts, "length": lengths}
    elif isinstance(data_link, h5py.Dataset):
        return np.arange(len(indices))


def write_links(
    file: File | Group,
    links: GalaxyPropertyLink | HaloPropertyLink,
    indices: np.ndarray,
):
    group = file.require_group("data_linked")
    for key, value in links.items():
        link_to_write = pack_links(value, indices)  # type: ignore
        if key == "sod_profile":
            group.create_dataset("sod_profile_idx", data=link_to_write)
        else:
            group.create_dataset(f"{key}_start", data=link_to_write["start_index"])
            group.create_dataset(f"{key}_size", data=link_to_write["length"])


class DataLink(TypedDict):
    start_index: np.ndarray  # The starting index of the link in the particle file
    length: np.ndarray  # The

class GravityOnlyHaloPropertyLink(TypedDict):
    dm_particles: DataLink
    halo_profiles: np.ndarray

class HaloPropertyLink(TypedDict):
    dm_particles: DataLink
    agn_particles: DataLink
    gas_particles: DataLink
    star_particles: DataLink
    galaxy_properties: DataLink
    halo_profiles: np.ndarray


class GalaxyPropertyLink(TypedDict):
    galaxy_particles: DataLink
