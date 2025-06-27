from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import h5py

from opencosmo import dataset as d
from opencosmo import io
from opencosmo import structure as s
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.structure.builder import build_dataset

LINK_ALIASES = {  # Left: Name in file, right: Name in collection
    "sodbighaloparticles_star_particles": "star_particles",
    "sodbighaloparticles_dm_particles": "dm_particles",
    "sodbighaloparticles_gravity_particles": "gravity_particles",
    "sodbighaloparticles_agn_particles": "agn_particles",
    "sodbighaloparticles_gas_particles": "gas_particles",
    "sod_profile": "halo_profiles",
    "galaxyproperties": "galaxy_properties",
    "galaxyparticles_star_particles": "star_particles",
}

ALLOWED_LINKS = {  # h5py.Files that can serve as a link holder and
    "halo_properties": ["halo_particles", "halo_profiles", "galaxy_properties"],
    "galaxy_properties": ["galaxy_particles"],
}


def get_link_spec(*headers: OpenCosmoHeader) -> dict[str, list[str]]:
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

    source_files = [dt for dt in data_types if dt in ALLOWED_LINKS]
    if not source_files:
        raise ValueError("No valid link source files found in headers")

    dtypes_to_headers = {header.file.data_type: header for header in headers}

    links = defaultdict(list)  # {file: [link_header, ...]}
    for file in source_files:
        for link in ALLOWED_LINKS[file]:
            try:
                link_header = dtypes_to_headers[link]
                # Check that the headers come from the same simulation
                if link_header.simulation != dtypes_to_headers[file].simulation:
                    raise ValueError(f"Simulation mismatch between {file} and {link}")
                links[file].append(link)
            except KeyError:
                continue  # No link header found for this file

    return links


def open_linked_files(*files: Path):
    """
    Open a collection of files that are linked together, such as a
    properties file and a particle file.

    """
    if len(files) == 1 and isinstance(files[0], list):
        return open_linked_files(*files[0])

    file_handles = [h5py.File(file, "r") for file in files]
    files_by_type = {f["header"]["file"].attrs["data_type"]: f for f in file_handles}
    headers = [read_header(file) for file in file_handles]
    headers_by_type = {header.file.data_type: header for header in headers}
    linked_files = get_link_spec(*headers)
    return build_structure_collection(linked_files, files_by_type, headers_by_type)


def open_linked_file(
    file_handle: h5py.File | h5py.Group,
    datasets_to_get: Optional[Iterable[str]] = None,
) -> s.StructureCollection:
    """
    Open a single file that contains both properties and linked datasets.
    """
    outputs = {}
    header = read_header(file_handle)
    datasets = set(k for k in file_handle.keys() if k != "header")
    # bespoke for now
    # Needs to be rewritten...
    if (
        "galaxy_properties" in file_handle.keys()
        and "data" not in file_handle["galaxy_properties"].keys()
    ):
        outputs["galaxy_properties"] = open_linked_file(
            file_handle["galaxy_properties"]
        )

    if "halo_properties" in file_handle.keys():
        source_dataset = io.open(file_handle["halo_properties"])
        if not isinstance(source_dataset, d.Dataset):
            raise ValueError("Expected dataset for link source!")
        datasets.remove("halo_properties")
        handlers = get_link_handlers(file_handle["halo_properties"], datasets, header)
        if "galaxy_properties" in outputs:
            datasets.remove("galaxy_properties")
        linked_datasets: dict[str, d.Dataset | s.StructureCollection] = {
            key: build_dataset(file_handle[key], header) for key in datasets
        }
        linked_datasets.update(outputs)
        collection = s.StructureCollection(
            source_dataset, header, linked_datasets, handlers
        )

    else:
        source_dataset = io.open(file_handle["galaxy_properties"])
        if not isinstance(source_dataset, d.Dataset):
            raise ValueError("Expected dataset for link source!")
        datasets.remove("galaxy_properties")
        handlers = get_link_handlers(
            file_handle["galaxy_properties"],
            datasets,
            header,
        )
        linked_datasets = {
            key: build_dataset(file_handle[key], header) for key in datasets
        }
        collection = s.StructureCollection(
            source_dataset, header, linked_datasets, handlers
        )

    return collection


def get_linked_datasets(
    linked_files_by_type: dict[str, h5py.File | h5py.Group],
    header: OpenCosmoHeader,
):
    datasets = {}
    for dtype, pointer in linked_files_by_type.items():
        if "data" not in pointer.keys():
            datasets.update(
                {
                    k: build_dataset(pointer[k], header)
                    for k in pointer.keys()
                    if k != "header"
                }
            )
        else:
            datasets.update({dtype: build_dataset(pointer, header)})
    return datasets


def build_structure_collection(
    link_spec: dict[str, list[str]],
    files_by_type: dict[str, h5py.File | h5py.Group],
    headers: dict[str, OpenCosmoHeader],
) -> s.StructureCollection:
    output: dict[str, s.StructureCollection] = {}
    source_names = set(link_spec.keys())
    while set(output.keys()) != source_names:
        for source, targets in link_spec.items():
            if source in output:
                continue
            if subcolls := set(targets).intersection(source_names):
                if not subcolls.issubset(set(output.keys())):
                    continue
            src_dataset = io.open(files_by_type[source])
            if not isinstance(src_dataset, d.Dataset):
                raise ValueError("Expected a dataset for the link source!")
            linked_datasets = get_linked_datasets(
                {t: files_by_type[t] for t in targets}, headers[source]
            )
            link_handlers = get_link_handlers(
                files_by_type[source], linked_datasets.keys(), headers[source]
            )
            for t in targets:
                if t in output:
                    linked_datasets[t] = output[t]
            output[source] = s.StructureCollection(
                src_dataset, headers[source], linked_datasets, link_handlers
            )
            final_source = source

    return output[final_source]


def get_link_handlers(
    link_file: h5py.File | h5py.Group,
    linked_files: Iterable[str],
    header: OpenCosmoHeader,
) -> dict[str, s.LinkedDatasetHandler]:
    if "data_linked" not in link_file.keys():
        raise KeyError("No linked datasets found in the file.")
    links = link_file["data_linked"]

    linked_files = list(linked_files)
    unique_dtypes = {key.rsplit("_", 1)[0] for key in links.keys()}
    output_links = {}
    for dtype in unique_dtypes:
        if dtype not in linked_files and LINK_ALIASES.get(dtype) not in linked_files:
            continue  # Skip if the linked file is not provided

        key = LINK_ALIASES.get(dtype, dtype)
        try:
            start = links[f"{dtype}_start"]
            size = links[f"{dtype}_size"]

            output_links[key] = s.LinkedDatasetHandler(
                (start, size),
            )
        except KeyError:
            index = links[f"{dtype}_idx"]
            output_links[key] = s.LinkedDatasetHandler(index)
    return output_links
