from __future__ import annotations

from collections import defaultdict
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Optional

import numpy as np

from opencosmo import dataset as d
from opencosmo import io
from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.structure import structure as sc

if TYPE_CHECKING:
    import h5py

    from opencosmo.io.iopen import FileTarget

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


def build_structure_collection(targets: list[FileTarget], ignore_empty: bool):
    link_sources: dict[str, list[io.iopen.DatasetTarget]] = defaultdict(list)
    link_targets: dict[str, dict[str, list[d.Dataset | sc.StructureCollection]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    dataset_targets: list[io.iopen.DatasetTarget] = reduce(
        lambda acc, t: acc + t["dataset_targets"], targets, []
    )
    for target in dataset_targets:
        if target["header"].file.data_type == "halo_properties":
            link_sources["halo_properties"].append(target)
        elif target["header"].file.data_type == "galaxy_properties":
            link_sources["galaxy_properties"].append(target)
        elif str(target["header"].file.data_type).startswith("halo"):
            dataset = io.iopen.open_single_dataset(
                target, bypass_lightcone=True, bypass_mpi=True
            )
            name = target["dataset_group"].name.split("/")[-1]
            if not name:
                name = target["header"].file.data_type
            elif name.startswith("halo_properties"):
                name = name[16:]
            link_targets["halo_properties"][name].append(dataset)
        elif str(target["header"].file.data_type).startswith("galaxy"):
            dataset = io.iopen.open_single_dataset(
                target, bypass_lightcone=True, bypass_mpi=True
            )
            name = target["dataset_group"].name.split("/")[-1]
            if not name:
                name = target["header"].file.data_type
            elif name.startswith("galaxy_properties"):
                name = name[18:]
            link_targets["galaxy_properties"][name].append(dataset)
        else:
            raise ValueError(
                f"Unknown data type for structure collection {target['header'].data_type}"
            )

    if (
        len(link_sources["halo_properties"]) > 1
        or len(link_sources["galaxy_properties"]) > 1
    ):
        # Potentially a lightcone structure collection
        return build_lightcone_structure_collection(link_sources, link_targets)

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


def build_lightcone_structure_collection(
    link_sources: dict[str, list[io.iopen.DatasetTarget]],
    link_targets: dict[str, dict[str, list[d.Dataset | sc.StructureCollection]]],
):
    found_redshift_steps: set[int] = set()
    for source_type, source_list in link_sources.items():
        if not all(t["header"].file.is_lightcone for t in source_list):
            raise ValueError("All sources must be lightcone datasets!")
        redshift_steps = set(t["header"].file.step for t in source_list)
        if found_redshift_steps and found_redshift_steps != redshift_steps:
            raise ValueError(
                "All source types must have the same set of redshift steps!"
            )
        if not all(
            t.header.file.is_lightcone
            for t in chain.from_iterable(link_targets[source_type].values())
        ):
            raise ValueError("All dataset must be lightcone datasets!")
        for targets in link_targets[source_type].values():
            target_redshift_steps = set(t.header.file.step for t in targets)
            if target_redshift_steps != redshift_steps:
                raise ValueError(
                    "All datasets must have the same set of redshift steps!"
                )
    if (
        len(link_sources.get("galaxy_properties", [])) > 0
        and "galaxy_properties" in link_targets
    ):
        # Galaxy properties and galaxy particles
        datasets = [
            io.iopen.open_single_dataset(
                t,
                "data_linked",
                bypass_lightcone=True,
                bypass_mpi=len(link_sources.get("halo_properties", [])) > 0,
            )
            for t in link_sources["galaxy_properties"]
        ]
        galaxy_lightcone = lc.Lightcone.from_datasets(
            {ds.header.file.step: ds for ds in datasets}
        )
        galaxy_target_datasets = {}
        for target_type, targets in link_targets["galaxy_properties"].items():
            galaxy_target_datasets[target_type] = lc.Lightcone.from_datasets(
                {ds.header.file.step: ds for ds in targets}  # type: ignore # already asserted this step exists
            )
        collection = sc.StructureCollection(galaxy_lightcone, galaxy_target_datasets)
        if len(link_sources.get("halo_properties", [])) > 0:
            link_targets["halo_properties"]["galaxy_properties"] = collection
        else:
            return collection

    source_list = link_sources["halo_properties"]
    source_datasets = [
        io.iopen.open_single_dataset(t, "data_linked", bypass_lightcone=True)
        for t in source_list
    ]
    source_lightcone = lc.Lightcone.from_datasets(
        {ds.header.file.step: ds for ds in source_datasets}
    )
    output_targets = {}
    for target_type, targets in link_targets[source_type].items():
        if isinstance(targets, (d.Dataset, sc.StructureCollection)):
            output_targets[target_type] = targets
            continue

        output_targets[target_type] = lc.Lightcone.from_datasets(
            {ds.header.file.step: ds for ds in targets}
        )
    return sc.StructureCollection(source_lightcone, output_targets)


def __build_structure_collection(
    halo_properties_target: Optional[io.iopen.DatasetTarget],
    galaxy_properties_target: Optional[io.iopen.DatasetTarget],
    link_targets: dict[str, dict[str, d.Dataset | sc.StructureCollection]],
    ignore_empty: bool,
):
    if galaxy_properties_target is not None and "galaxy_properties" in link_targets:
        # Galaxy properties and galaxy particles
        source_dataset = io.iopen.open_single_dataset(
            galaxy_properties_target,
            metadata_group="data_linked",
            bypass_lightcone=True,
            bypass_mpi=halo_properties_target is not None,
        )
        if ignore_empty and halo_properties_target is None:
            source_dataset = remove_empty(source_dataset)
        collection = sc.StructureCollection(
            source_dataset,
            link_targets["galaxy_properties"],
        )
        if halo_properties_target is not None:
            link_targets["halo_properties"]["galaxy_properties"] = collection
        else:
            return collection

    if (
        halo_properties_target is not None
        and galaxy_properties_target is not None
        and "galaxy_properties" not in link_targets
    ):
        # Halo properties and galaxy properties, but no galaxy particles
        galaxy_properties = io.iopen.open_single_dataset(
            galaxy_properties_target, bypass_lightcone=True, bypass_mpi=True
        )
        link_targets["halo_properties"]["galaxy_properties"] = galaxy_properties

    if halo_properties_target is not None and link_targets["halo_properties"]:
        source_dataset = io.iopen.open_single_dataset(
            halo_properties_target, metadata_group="data_linked", bypass_lightcone=True
        )
        if ignore_empty:
            source_dataset = remove_empty(source_dataset)

        return sc.StructureCollection(
            source_dataset,
            link_targets["halo_properties"],
        )
