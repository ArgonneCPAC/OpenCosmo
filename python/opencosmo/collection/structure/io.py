from __future__ import annotations

from collections import defaultdict
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from opencosmo import io
from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone import state as lcst
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.structure import links
from opencosmo.collection.structure import state as scst
from opencosmo.collection.structure.links import LINK_ALIASES
from opencosmo.collection.structure.state import StructureState
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.state import DatasetState
from opencosmo.io.index_spec import index_spec_for
from opencosmo.io.updaters import do_idx_update, do_start_update
from opencosmo.mapping.read import read_link_set

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from uuid import UUID

    import h5py

    from opencosmo.collection.structure.state import Source, StructureChild
    from opencosmo.io.discover import LinkLayout
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.mapping.mapping import DatasetMatchSet

    # Link targets resolved against one source dataset (one lightcone step).
    StepTarget = DatasetState | StructureState

ALLOWED_LINKS = {  # h5py.Files that can serve as a link holder and
    "halo_properties": ["halo_particles", "halo_profiles", "galaxy_properties"],
    "galaxy_properties": ["galaxy_particles"],
}


def resolve_link_targets(
    layout: LinkLayout,
    targets: Mapping[str, StepTarget],
) -> tuple[dict[str, UUID], dict[str, UUID]]:
    """Resolve on-disk link prefixes against opened targets.

    Returns ``(prefix_to_uuid, name_to_uuid)``: the first keyed by on-disk link
    prefix for ``read_link_set``, the second by collection-facing name for
    ``DatasetMatchSet.with_aliases``. A nested galaxy collection is still keyed
    as "galaxy_properties" here; ``StructureState.from_parts`` renames it.
    """
    prefix_to_uuid: dict[str, UUID] = {}
    name_to_uuid: dict[str, UUID] = {}
    for slot in layout.slots:
        name = LINK_ALIASES[slot.prefix]
        target = targets.get(name)
        if target is None:
            continue
        if isinstance(target, StructureState):
            source = target.source
            assert isinstance(source, DatasetState)
            target = source
        prefix_to_uuid[slot.prefix] = target.uuid
        name_to_uuid[name] = target.uuid
    return prefix_to_uuid, name_to_uuid


def build_match_sets(
    source_target: DatasetTarget,
    source: DatasetState,
    targets: Mapping[str, StepTarget],
) -> dict[UUID, DatasetMatchSet]:
    """Build the resolved link match set for one properties source dataset."""
    # Discovery sets link_layout if and only if the group has a /data_linked group,
    # so the layout alone answers whether there are links to resolve.
    layout = source_target.link_layout
    if layout is None:
        return {}
    prefix_to_uuid, name_to_uuid = resolve_link_targets(layout, targets)
    match_set = read_link_set(
        source_target.group["data_linked"],
        layout,
        source_target.uuid,
        prefix_to_uuid,
    )
    if match_set is None:
        return {}
    return {source.uuid: match_set.with_aliases(name_to_uuid)}


def __step_targets(
    targets: Mapping[str, StructureChild], step: int
) -> dict[str, StepTarget]:
    resolved: dict[str, StepTarget] = {}
    for name, target in targets.items():
        if isinstance(target, LightconeState):
            step_target = lcst.view(target, step)
            assert isinstance(step_target, DatasetState)
            resolved[name] = step_target
        elif isinstance(target, StructureState) and isinstance(
            target.source, LightconeState
        ):
            step_source = lcst.view(target.source, step)
            assert isinstance(step_source, DatasetState)
            resolved[name] = step_source
        else:
            resolved[name] = target
    return resolved


def __build_lightcone_match_sets(
    source_targets: Iterable[DatasetTarget],
    source_datasets: Iterable[DatasetState],
    targets: Mapping[str, StructureChild],
) -> dict[UUID, DatasetMatchSet]:
    sources = tuple(source_datasets)
    targets_by_step = {
        step: __step_targets(targets, step)
        for step in {
            ds.header.file.step for ds in sources if ds.header.file.step is not None
        }
    }
    match_sets: dict[UUID, DatasetMatchSet] = {}
    for source_target, source in zip(source_targets, sources, strict=True):
        step = source.header.file.step
        assert step is not None
        match_sets.update(
            build_match_sets(
                source_target,
                source,
                targets_by_step[step],
            )
        )
    return match_sets


def remove_empty(
    source: Source,
    match_sets: Mapping[UUID, DatasetMatchSet],
    opened_datasets: Iterable[str] | None = None,
) -> Source:
    """
    Drop structures that are empty in the linked datasets that were actually
    opened. When a user opens, say, particles and profiles together, they should
    be able to assume every structure has both -- the source data keeps particles
    and profiles separately, and this reproduces that "all present" guarantee. The
    ignore_empty flag on open() exists to override it.

    Only the link columns belonging to opened datasets are considered. The source
    metadata always carries link columns for every data type in the file (e.g.
    galaxy or particle links), so restricting to opened datasets avoids dropping
    structures based on links the user never asked for.
    """
    names = set(links.link_names(match_sets))
    if opened_datasets is not None:
        names &= set(opened_datasets)
    if not names:
        return source

    mask = np.ones(len(source), dtype=bool)
    for name in sorted(names):
        values, is_chunked = links.slot_values(match_sets, source, name)
        mask &= values != 0 if is_chunked else values != -1

    if mask.all():
        return source
    rows = np.where(mask)[0]
    if isinstance(source, DatasetState):
        return dsops.take_rows(source, rows)
    return lcops.take_rows(source, rows)


def validate_linked_groups(groups: dict[str, h5py.Group]):
    if "halo_properties" in groups:
        if "data_linked" not in groups["halo_properties"]:
            raise ValueError(
                "File appears to be a structure collection, but does not have links!"
            )
    elif (
        "galaxy_properties" in groups
        and "data_linked" not in groups["galaxy_properties"]
    ):
        raise ValueError(
            "File appears to be a structure collection, but does not have links!"
        )
    if len(groups) == 1:
        raise ValueError("Structure collections must have more than one dataset")


def __link_target_name(target: io.iopen.DatasetTarget, prefix: str) -> str:
    """Collection-facing name for one linked dataset.

    Particles and profiles are written one level deeper than the properties group
    they hang off, so their own path segment is not the name — their parent's is.
    Both are pure functions of the discovered group path; nothing here touches h5py.
    """
    name = target.name
    if "particles" in target.parent_name or "profiles" in target.parent_name:
        name = target.parent_name.rsplit("/", 1)[-1]

    if not name:
        return str(target.header.file.data_type)
    if name.startswith(prefix):
        return name[len(prefix) + 1 :]
    return name


def __open_state(
    target: DatasetTarget, index_kind: str, is_empty_ref: bool, *, is_source: bool
) -> DatasetState:
    spec = index_spec_for(index_kind, is_empty_ref, is_source=is_source)
    return io.iopen.open_dataset(target, spec)


def __lightcone_of(datasets: Iterable[DatasetState]) -> LightconeState:
    """Stack per-step datasets into a lightcone, running the lightcone open hooks."""
    leaves = []
    for dataset in datasets:
        step = dataset.header.file.step
        assert step is not None
        leaves.append(((step,), dataset))
    return lcops.from_dataset_states(leaves)


def build_structure_collection(
    targets: list[io.iopen.DatasetTarget],
    ignore_empty: bool,
    index_kind: str = "none",
    is_empty_ref: bool = False,
) -> StructureState:
    link_sources: dict[str, list[io.iopen.DatasetTarget]] = defaultdict(list)
    link_targets: dict[str, dict[str, list[DatasetState]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for target in targets:
        data_type = str(target.header.file.data_type)
        if data_type == "halo_properties":
            link_sources["halo_properties"].append(target)
        elif data_type == "galaxy_properties":
            link_sources["galaxy_properties"].append(target)
        elif data_type.startswith(("halo", "galaxy")):
            source_type = (
                "halo_properties"
                if data_type.startswith("halo")
                else "galaxy_properties"
            )
            dataset = __open_state(target, index_kind, is_empty_ref, is_source=False)
            name = __link_target_name(target, source_type)
            link_targets[source_type][name].append(dataset)
        else:
            raise ValueError(f"Unknown data type for structure collection {data_type}")

    if (
        index_kind == "redshift_step"
        or len(link_sources["halo_properties"]) > 1
        or len(link_sources["galaxy_properties"]) > 1
    ):
        # Lightcone structure collection. Under redshift-split we ALWAYS route here,
        # even when this rank holds a single step (one source per type), so every
        # rank builds the same collection kind and stays MPI-lockstep on the write
        # path.
        return build_lightcone_structure_collection(
            link_sources,
            link_targets,
            ignore_empty,
            index_kind=index_kind,
            is_empty_ref=is_empty_ref,
        )

    halo_properties_target = None
    galaxy_properties_target = None
    if link_sources["halo_properties"]:
        halo_properties_target = link_sources["halo_properties"][0]
    if link_sources["galaxy_properties"]:
        galaxy_properties_target = link_sources["galaxy_properties"][0]

    input_link_targets: dict[str, dict[str, StructureChild]] = defaultdict(dict)
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
        index_kind=index_kind,
        is_empty_ref=is_empty_ref,
    )


def build_lightcone_structure_collection(
    link_sources: dict[str, list[io.iopen.DatasetTarget]],
    link_targets: dict[str, dict[str, list[DatasetState]]],
    ignore_empty: bool = True,
    index_kind: str = "none",
    is_empty_ref: bool = False,
) -> StructureState:
    found_redshift_steps: set[int] = set()
    for source_type, source_list in link_sources.items():
        if not all(t.header.file.is_lightcone for t in source_list):
            raise ValueError("All sources must be lightcone datasets!")
        redshift_steps = {t.header.file.step for t in source_list}
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
            target_redshift_steps = {t.header.file.step for t in targets}
            if target_redshift_steps != redshift_steps:
                raise ValueError(
                    "All datasets must have the same set of redshift steps!"
                )

    # Linked children of the halo source that are already built, rather than
    # lists of per-step datasets.
    built_halo_targets: dict[str, StructureChild] = {}

    # NOTE: link_targets is a defaultdict, so accessing link_targets[source_type]
    # in the validation loop above may have created an empty "galaxy_properties"
    # entry. Use a truthy check (non-empty dict) rather than `in` so that the
    # "galaxy properties but no galaxy particles" case does not fall into the
    # galaxy-particles branch below.
    if len(link_sources.get("galaxy_properties", [])) > 0 and link_targets.get(
        "galaxy_properties"
    ):
        # Galaxy properties and galaxy particles
        is_galaxy_source = len(link_sources.get("halo_properties", [])) == 0
        galaxy_datasets = [
            __open_state(t, index_kind, is_empty_ref, is_source=is_galaxy_source)
            for t in link_sources["galaxy_properties"]
        ]
        galaxy_lightcone = __lightcone_of(galaxy_datasets)
        galaxy_targets: dict[str, StructureChild] = {
            target_type: __lightcone_of(targets)
            for target_type, targets in link_targets["galaxy_properties"].items()
        }
        galaxy_match_sets = __build_lightcone_match_sets(
            link_sources["galaxy_properties"],
            galaxy_datasets,
            galaxy_targets,
        )
        if len(link_sources.get("halo_properties", [])) > 0:
            built_halo_targets["galaxy_properties"] = scst.from_parts(
                galaxy_lightcone, galaxy_targets, galaxy_match_sets, resolve=True
            )
        else:
            galaxy_source: Source = galaxy_lightcone
            if ignore_empty:
                galaxy_source = remove_empty(
                    galaxy_lightcone, galaxy_match_sets, galaxy_targets.keys()
                )
            return scst.from_parts(
                galaxy_source, galaxy_targets, galaxy_match_sets, resolve=True
            )

    elif (
        len(link_sources.get("halo_properties", [])) > 0
        and len(link_sources.get("galaxy_properties", [])) > 0
    ):
        # Halo properties and galaxy properties, but no galaxy particles. Attach
        # the galaxy properties as a plain per-step linked dataset under the
        # halos, exactly like halo profiles.
        link_targets["halo_properties"]["galaxy_properties"] = [
            __open_state(t, index_kind, is_empty_ref, is_source=False)
            for t in link_sources["galaxy_properties"]
        ]

    halo_source_list = link_sources["halo_properties"]
    halo_datasets = [
        __open_state(t, index_kind, is_empty_ref, is_source=True)
        for t in halo_source_list
    ]
    source: Source = __lightcone_of(halo_datasets)

    # Iterate the linked type names in a deterministic (sorted) order so that
    # make_schema's local, union-free iteration produces the same child ordering
    # on every rank under redshift-split -- defensive lockstep for the mixed-type
    # write path.
    output_targets: dict[str, StructureChild] = {}
    for target_type in sorted(
        link_targets["halo_properties"].keys() | built_halo_targets.keys()
    ):
        if target_type in built_halo_targets:
            output_targets[target_type] = built_halo_targets[target_type]
        else:
            output_targets[target_type] = __lightcone_of(
                link_targets["halo_properties"][target_type]
            )
    halo_match_sets = __build_lightcone_match_sets(
        halo_source_list, halo_datasets, output_targets
    )
    if ignore_empty:
        source = remove_empty(source, halo_match_sets, output_targets.keys())
    return scst.from_parts(source, output_targets, halo_match_sets, resolve=True)


def __build_structure_collection(
    halo_properties_target: io.iopen.DatasetTarget | None,
    galaxy_properties_target: io.iopen.DatasetTarget | None,
    link_targets: dict[str, dict[str, StructureChild]],
    ignore_empty: bool,
    index_kind: str = "none",
    is_empty_ref: bool = False,
) -> StructureState:
    if galaxy_properties_target is not None and "galaxy_properties" in link_targets:
        # Galaxy properties and galaxy particles
        galaxy_source: Source = __open_state(
            galaxy_properties_target,
            index_kind,
            is_empty_ref,
            is_source=halo_properties_target is None,
        )
        assert isinstance(galaxy_source, DatasetState)
        galaxy_match_sets = build_match_sets(
            galaxy_properties_target,
            galaxy_source,
            __dataset_targets(link_targets["galaxy_properties"]),
        )
        if ignore_empty and halo_properties_target is None:
            galaxy_source = remove_empty(
                galaxy_source,
                galaxy_match_sets,
                link_targets["galaxy_properties"].keys(),
            )
        collection = scst.from_parts(
            galaxy_source,
            link_targets["galaxy_properties"],
            galaxy_match_sets,
            resolve=True,
        )
        if halo_properties_target is None:
            return collection
        link_targets["halo_properties"]["galaxy_properties"] = collection

    if (
        halo_properties_target is not None
        and galaxy_properties_target is not None
        and "galaxy_properties" not in link_targets
    ):
        # Halo properties and galaxy properties, but no galaxy particles
        link_targets["halo_properties"]["galaxy_properties"] = __open_state(
            galaxy_properties_target, index_kind, is_empty_ref, is_source=False
        )

    if halo_properties_target is None or not link_targets["halo_properties"]:
        raise RuntimeError("Unable to build a structure collection from these files")

    source: Source = __open_state(
        halo_properties_target, index_kind, is_empty_ref, is_source=True
    )
    assert isinstance(source, DatasetState)
    halo_match_sets = build_match_sets(
        halo_properties_target,
        source,
        __dataset_targets(link_targets["halo_properties"]),
    )
    if ignore_empty:
        source = remove_empty(
            source, halo_match_sets, link_targets["halo_properties"].keys()
        )
    return scst.from_parts(
        source, link_targets["halo_properties"], halo_match_sets, resolve=True
    )


def __dataset_targets(targets: Mapping[str, StructureChild]) -> dict[str, StepTarget]:
    """Snapshot link targets: datasets and nested collections, never lightcones."""
    resolved: dict[str, StepTarget] = {}
    for name, target in targets.items():
        assert not isinstance(target, LightconeState)
        resolved[name] = target
    return resolved


def rebuild_data_linked(source_schema):
    """Attach MPI link-coordinate updates to a structure source schema."""
    if (
        source_schema.type == io.schema.FileEntry.LIGHTCONE
        and "data" not in source_schema.children
    ):
        for key, value in source_schema.children.items():
            source_schema.children[key] = rebuild_data_linked(value)
        return source_schema

    for colname, column in source_schema.children["data_linked"].columns.items():
        if "idx" in colname:
            column.set_transformation(do_idx_update)
        elif "start" in colname:
            size_colname = colname.replace("start", "size")
            size_data = source_schema.children["data_linked"].columns[size_colname].data
            updater = partial(do_start_update, size=size_data)
            column.set_transformation(updater)
    return source_schema
