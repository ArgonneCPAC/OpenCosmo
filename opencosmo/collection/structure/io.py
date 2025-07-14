from collections import defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from deprecated import deprecated

from opencosmo import dataset as d
from opencosmo import io
from opencosmo.collection.structure import collection as sc
from opencosmo.dataset.build import build_dataset
from opencosmo.header import OpenCosmoHeader
from opencosmo.index import DataIndex

from .handler import LinkedDatasetHandler

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


@deprecated(
    version="0.8",
    reason="oc.open_linked_files is deprecated and will be removed in version 1.0. "
    "Please use oc.open instead",
)
def open_linked_files(*files: Path, **load_kwargs: bool):
    """
    **WARNING: THIS METHOD IS DEPCREATED AND WILL BE REMOVED IN A FUTURE
    VERSION. PLEASE USE** :py:meth:`opencosmo.open`

    Open a collection of files that are linked together, such as a
    properties file and a particle file.

    """
    if len(files) == 1 and isinstance(files[0], list):
        return open_linked_files(*files[0])

    return io.io.open(*files, **load_kwargs)


def validate_linked_groups(groups: dict[str, h5py.Group]):
    if "halo_properties" in groups:
        if "data_linked" not in groups["halo_properties"].keys():
            raise ValueError(
                "File appears to be a structure collection, but does not have links!"
            )
    elif "galaxy_properties" in groups:
        if "data_linked" not in groups["galaxy_properties"].keys():
            raise ValueError(
                "File appears to be a structure collection, but does not have links!"
            )
    if len(groups) == 1:
        raise ValueError("Structure collections must have more than one dataset")


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


def make_index_with_linked_data(
    index: DataIndex, links: dict[str, LinkedDatasetHandler]
):
    mask = np.ones(len(index), dtype=bool)
    for link in links.values():
        mask &= link.has_linked_data(index)

    return index.mask(mask)


def build_structure_collection(targets: list[io.io.OpenTarget], ignore_empty: bool):
    link_sources = defaultdict(list)
    link_targets: dict[str, dict[str, d.Dataset | sc.StructureCollection]] = (
        defaultdict(dict)
    )
    for target in targets:
        if target.data_type == "halo_properties":
            link_sources["halo_properties"].append(target)
        elif target.data_type == "galaxy_properties":
            link_sources["galaxy_properties"].append(target)
        elif target.data_type.startswith("halo"):
            dataset = io.io.open_single_dataset(target)
            name = target.group.name.split("/")[-1]
            if not name:
                name = target.data_type
            elif name.startswith("halo_properties"):
                name = name[16:]
            link_targets["halo_targets"][name] = dataset
        elif target.data_type.startswith("galaxy"):
            dataset = io.io.open_single_dataset(target)
            name = target.group.name.split("/")[-1]
            if not name:
                name = target.data_type
            elif name.startswith("galaxy_properties"):
                name = name[18:]
            link_targets["galaxy_targets"][name] = dataset
        else:
            raise ValueError(
                f"Unknown data type for structure collection {target.data_type}"
            )

    if len(link_sources["galaxy_properties"]) == 1 and link_targets["galaxy_targets"]:
        handlers = get_link_handlers(
            link_sources["galaxy_properties"][0].group,
            list(link_targets["galaxy_targets"].keys()),
            link_sources["galaxy_properties"][0].header,
        )

        source_dataset = io.io.open_single_dataset(link_sources["galaxy_properties"][0])
        if ignore_empty:
            new_index = make_index_with_linked_data(source_dataset.index, handlers)
            source_dataset = source_dataset.with_index(new_index)
        collection = sc.StructureCollection(
            source_dataset,
            source_dataset.header,
            link_targets["galaxy_targets"],
            handlers,
        )
        if len(link_sources["halo_properties"]) != 0:
            link_targets["halo_targets"]["galaxy_properties"] = collection
        else:
            return collection

    if len(link_sources["halo_properties"]) == 1 and link_targets["halo_targets"]:
        handlers = get_link_handlers(
            link_sources["halo_properties"][0].group,
            list(link_targets["halo_targets"].keys()),
            link_sources["halo_properties"][0].header,
        )
        source_dataset = io.io.open_single_dataset(link_sources["halo_properties"][0])

        if ignore_empty:
            new_index = make_index_with_linked_data(source_dataset.index, handlers)
            source_dataset = source_dataset.with_index(new_index)

        return sc.StructureCollection(
            source_dataset,
            source_dataset.header,
            link_targets["halo_targets"],
            handlers,
        )


def get_link_handlers(
    link_file: h5py.File | h5py.Group,
    linked_files: Iterable[str],
    header: OpenCosmoHeader,
) -> dict[str, LinkedDatasetHandler]:
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

            output_links[key] = LinkedDatasetHandler(
                (start, size),
            )
        except KeyError:
            index = links[f"{dtype}_idx"]
            output_links[key] = LinkedDatasetHandler(index)
    return output_links
