from __future__ import annotations

from collections import defaultdict
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Optional

import numpy as np

import opencosmo as oc
from opencosmo import dataset as d
from opencosmo import io
from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.structure import structure as sc
from opencosmo.collection.structure.handler import LINK_ALIASES

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


def _apply_offset_corrections(
    source_by_step: dict[int, d.Dataset],
    targets_by_step: dict[str, dict[int, d.Dataset | sc.StructureCollection]],
) -> dict[int, d.Dataset]:
    """
    Correct step-local _start and _idx metadata columns to be globally correct
    before stacking per-step source datasets into a Lightcone.

    For _start columns: apply a lazy DerivedColumn offset (oc.col(name) + offset).
    For _idx columns: apply the offset eagerly (only to non-negative values).

    targets_by_step keys may be either file-level group name prefixes (e.g.
    "sodbighaloparticles_dm_particles") or alias values (e.g. "galaxy_properties").
    Column names always use file-level prefixes (e.g. "sodbighaloparticles_dm_particles_start"),
    so we match via direct lookup then fall back to a LINK_ALIASES alias lookup.
    """
    steps = sorted(source_by_step)

    type_step_offset: dict[str, dict[int, int]] = {}
    for target_type, step_map in targets_by_step.items():
        cumulative = 0
        per_step: dict[int, int] = {}
        for step in steps:
            per_step[step] = cumulative
            ds = step_map.get(step)
            if ds is not None:
                cumulative += len(ds)
        type_step_offset[target_type] = per_step

    corrected: dict[int, d.Dataset] = {}
    for step in steps:
        step_ds = source_by_step[step]
        meta_cols = set(step_ds.meta_columns)
        updates: dict = {}

        for col in meta_cols:
            if col.endswith("_start"):
                prefix = col[:-6]
                is_idx = False
            elif col.endswith("_idx"):
                prefix = col[:-4]
                is_idx = True
            else:
                continue

            # Try direct match (target_type key == file-level prefix), then
            # fall back to alias lookup for cases like "galaxyproperties" -> "galaxy_properties".
            offset = None
            if prefix in type_step_offset:
                offset = type_step_offset[prefix][step]
            elif prefix in LINK_ALIASES and LINK_ALIASES[prefix] in type_step_offset:
                offset = type_step_offset[LINK_ALIASES[prefix]][step]

            if not offset:
                continue

            if is_idx:
                arr = step_ds.get_metadata([col])[col].copy()
                arr[arr >= 0] += offset
                updates[col] = arr
            else:
                updates[col] = oc.col(col) + offset

        if updates:
            step_ds = step_ds.with_new_columns(allow_overwrite=True, **updates)
        corrected[step] = step_ds

    return corrected


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
        galaxy_datasets = [
            io.iopen.open_single_dataset(
                t,
                "data_linked",
                bypass_lightcone=True,
                bypass_mpi=len(link_sources.get("halo_properties", [])) > 0,
            )
            for t in link_sources["galaxy_properties"]
        ]
        galaxy_source_by_step = {ds.header.file.step: ds for ds in galaxy_datasets}
        galaxy_targets_by_step = {
            target_type: {ds.header.file.step: ds for ds in targets}  # type: ignore
            for target_type, targets in link_targets["galaxy_properties"].items()
        }
        galaxy_source_by_step = _apply_offset_corrections(
            galaxy_source_by_step, galaxy_targets_by_step
        )
        galaxy_lightcone = lc.Lightcone.from_datasets(galaxy_source_by_step)
        galaxy_target_datasets = {}
        for target_type, targets in link_targets["galaxy_properties"].items():
            galaxy_target_datasets[target_type] = lc.Lightcone.from_datasets(
                {ds.header.file.step: ds for ds in targets}  # type: ignore
            )
        collection = sc.StructureCollection(galaxy_lightcone, galaxy_target_datasets)
        if len(link_sources.get("halo_properties", [])) > 0:
            link_targets["halo_properties"]["galaxy_properties"] = collection
        else:
            return collection

    halo_source_list = link_sources["halo_properties"]
    halo_datasets = [
        io.iopen.open_single_dataset(t, "data_linked", bypass_lightcone=True)
        for t in halo_source_list
    ]
    halo_source_by_step = {ds.header.file.step: ds for ds in halo_datasets}
    halo_targets_by_step: dict[str, dict[int, d.Dataset]] = {}
    for target_type, targets in link_targets["halo_properties"].items():
        if isinstance(targets, sc.StructureCollection):
            # For a nested SC (e.g. galaxies), target_type is its source dtype
            # ("galaxy_properties"), so targets[target_type] returns the source
            # Lightcone, giving us per-step sizes for offset accounting.
            inner_lc = targets[target_type]
            assert isinstance(inner_lc, lc.Lightcone)
            halo_targets_by_step[target_type] = dict(inner_lc)
        elif isinstance(targets, list):
            halo_targets_by_step[target_type] = {
                ds.header.file.step: ds for ds in targets
            }
    halo_source_by_step = _apply_offset_corrections(
        halo_source_by_step, halo_targets_by_step
    )
    source_lightcone = lc.Lightcone.from_datasets(halo_source_by_step)

    output_targets = {}
    for target_type, targets in link_targets["halo_properties"].items():
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
