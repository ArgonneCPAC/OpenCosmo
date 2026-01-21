from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import numpy as np

from opencosmo import io
from opencosmo.collection import lightcone as lc
from opencosmo.collection.structure import structure as sc

if TYPE_CHECKING:
    import h5py

    from opencosmo import dataset as d
    from opencosmo.header import OpenCosmoHeader

ALLOWED_LINKS = {  # h5py.Files that can serve as a link holder and
    "halo_properties": ["halo_particles", "halo_profiles", "galaxy_properties"],
    "galaxy_properties": ["galaxy_particles"],
}


def remove_empty(dataset):
    metadata = dataset.get_metadata()
    mask = np.ones(len(dataset), dtype=bool)
    for name, col in metadata.items():
        if "size" in name:
            mask &= col != 0
        elif "idx" in name:
            mask &= col != -1

    if not mask.all():
        dataset = dataset.take_rows(np.where(mask)[0])
    return dataset


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
    targets = {}
    for dtype, pointer in linked_files_by_type.items():
        if "data" not in pointer.keys():
            targets.update(
                {
                    k: io.io.OpenTarget(pointer[k], header)
                    for k in pointer.keys()
                    if k != "header"
                }
            )
        else:
            targets.update({dtype: io.io.OpenTarget(pointer, header)})
    datasets = {
        dtype: io.io.open_single_dataset(target, bypass_lightcone=True, bypass_mpi=True)
        for dtype, target in targets.items()
    }
    return datasets


def build_structure_collection(targets: list[io.io.OpenTarget], ignore_empty: bool):
    link_sources = defaultdict(list)
    link_targets: dict[str, dict[str, list[d.Dataset | sc.StructureCollection]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for target in targets:
        if target.data_type == "halo_properties":
            link_sources["halo_properties"].append(target)
        elif target.data_type == "galaxy_properties":
            link_sources["galaxy_properties"].append(target)
        elif target.data_type.startswith("halo"):
            dataset = io.io.open_single_dataset(
                target, bypass_lightcone=True, bypass_mpi=True
            )
            name = target.group.name.split("/")[-1]
            if not name:
                name = target.data_type
            elif name.startswith("halo_properties"):
                name = name[16:]
            link_targets["halo_targets"][name].append(dataset)
        elif target.data_type.startswith("galaxy"):
            dataset = io.io.open_single_dataset(
                target, bypass_lightcone=True, bypass_mpi=True
            )
            name = target.group.name.split("/")[-1]
            if not name:
                name = target.data_type
            elif name.startswith("galaxy_properties"):
                name = name[18:]
            link_targets["galaxy_targets"][name].append(dataset)
        else:
            raise ValueError(
                f"Unknown data type for structure collection {target.data_type}"
            )

    if (
        len(link_sources["halo_properties"]) > 1
        or len(link_sources["galaxy_properties"]) > 1
    ):
        raise NotImplementedError(
            "Opening structure collections that span multiple redshifts is not currently supported"
        )
        # Potentially a lightcone structure collection
        collections = {}
        sources_by_step, targets_by_step = __sort_by_step(link_sources, link_targets)
        if set(sources_by_step.keys()) != set(targets_by_step.keys()):
            raise ValueError("Datasets are not the same across all lightcone steps!")
        for step, sources in sources_by_step.items():
            halo_properties = sources.get("halo_properties")
            galaxy_properties = sources.get("galaxy_properties")
            targets = targets_by_step[step]
            collection = __build_structure_collection(
                halo_properties, galaxy_properties, targets, ignore_empty
            )
            collections[step] = collection

        expected_datasets = set(next(iter(collections.values())).keys())
        for collection in collections.values():
            if set(collection.keys()) != expected_datasets:
                raise ValueError(
                    "All structure collections in a lightcone must have the same set of datasets"
                )
        return lc.Lightcone(collections)

    halo_properties_target = None
    galaxy_properties_target = None
    if link_sources["halo_properties"]:
        halo_properties_target = link_sources["halo_properties"][0]
    if link_sources["galaxy_properties"]:
        galaxy_properties_target = link_sources["galaxy_properties"][0]

    input_link_targets: dict[str, dict[str, d.Dataset | sc.StructureCollection]] = (
        defaultdict(dict)
    )
    for source_type, source_targets in link_targets.items():
        if any(len(ts) > 1 for ts in source_targets.values()):
            raise ValueError("Found more than one linked file of a given type!")
        input_link_targets[source_type] = {
            key: t[0] for key, t in source_targets.items()
        }

    return __build_structure_collection(
        halo_properties_target,
        galaxy_properties_target,
        input_link_targets,
        ignore_empty,
    )


def __sort_by_step(link_sources: dict[str, list[io.io.OpenTarget]], link_targets):
    sources_by_step: dict[int, dict[str, io.io.OpenTarget]] = defaultdict(dict)
    targets_by_step: dict[int, dict[str, dict[str, d.Dataset]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for source_name, sources in link_sources.items():
        for source in sources:
            if not source.header.file.is_lightcone:
                raise ValueError(
                    "Recived multiple source datasets of a single type, but not all are lightcone datasets!"
                )
            if source.header.file.step is None:
                raise ValueError("No step in source!")

            sources_by_step[source.header.file.step][source_name] = source
    for target_type, targets_ in link_targets.items():
        for target_name, targets in targets_.items():
            for target in targets:
                if not target.header.file.is_lightcone:
                    raise ValueError(
                        "Recived multiple datasets of a single type, but not all are lightcone datasets!"
                    )
                targets_by_step[target.header.file.step][target_type][target_name] = (
                    target
                )

    return sources_by_step, targets_by_step


def __build_structure_collection(
    halo_properties_target: Optional[io.io.OpenTarget],
    galaxy_properties_target: Optional[io.io.OpenTarget],
    link_targets: dict[str, dict[str, d.Dataset | sc.StructureCollection]],
    ignore_empty: bool,
):
    if galaxy_properties_target is not None and "galaxy_targets" in link_targets:
        # Galaxy properties and galaxy particles
        source_dataset = io.io.open_single_dataset(
            galaxy_properties_target,
            metadata_group="data_linked",
            bypass_lightcone=True,
            bypass_mpi=halo_properties_target is not None,
        )
        if ignore_empty and halo_properties_target is None:
            source_dataset = remove_empty(source_dataset)
        collection = sc.StructureCollection(
            source_dataset,
            source_dataset.header,
            link_targets["galaxy_targets"],
        )
        if halo_properties_target is not None:
            link_targets["halo_targets"]["galaxy_properties"] = collection
        else:
            return collection

    if (
        halo_properties_target is not None
        and galaxy_properties_target is not None
        and "galaxy_targets" not in link_targets
    ):
        # Halo properties and galaxy properties, but no galaxy particles
        galaxy_properties = io.io.open_single_dataset(
            galaxy_properties_target, bypass_lightcone=True, bypass_mpi=True
        )
        link_targets["halo_targets"]["galaxy_properties"] = galaxy_properties

    if halo_properties_target is not None and link_targets["halo_targets"]:
        source_dataset = io.io.open_single_dataset(
            halo_properties_target, metadata_group="data_linked", bypass_lightcone=True
        )
        if ignore_empty:
            source_dataset = remove_empty(source_dataset)

        return sc.StructureCollection(
            source_dataset,
            source_dataset.header,
            link_targets["halo_targets"],
        )
