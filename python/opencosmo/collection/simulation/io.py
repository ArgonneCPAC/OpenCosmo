from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from opencosmo.index import into_array
from opencosmo.io.schema import FileEntry, Schema, get_dataset_schema_index
from opencosmo.io.writer import ColumnWriter
from opencosmo.mpi import (
    gather_index,
    get_all_entries,
    redistribute_data,
    scatter_index,
)

if TYPE_CHECKING:
    from opencosmo.mpi import MPI


@dataclass(frozen=True)
class DatasetOutputLookup:
    """Global output coordinates and writer ownership for one dataset.

    ``raw_ids`` holds the globally sorted, unique raw row IDs, so a raw ID's
    output position is its position in that array. ``writer_ranks`` is aligned
    with ``raw_ids`` and names the rank that writes each row.
    """

    raw_ids: np.ndarray
    writer_ranks: np.ndarray


def __lookup_positions(lookup: DatasetOutputLookup, raw_ids: np.ndarray) -> np.ndarray:
    """Resolve raw row IDs to output positions, or -1 where they are absent."""
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    if len(lookup.raw_ids) == 0:
        return np.full(len(raw_ids), -1, dtype=np.int64)

    positions = np.searchsorted(lookup.raw_ids, raw_ids)
    np.clip(positions, 0, len(lookup.raw_ids) - 1, out=positions)
    found = lookup.raw_ids[positions] == raw_ids
    return np.where(found, positions, -1).astype(np.int64)


def __lookup_ranks(lookup: DatasetOutputLookup, positions: np.ndarray) -> np.ndarray:
    """Map resolved output positions to their writer ranks, preserving -1."""
    found = positions >= 0
    ranks = np.full(len(positions), -1, dtype=np.int64)
    ranks[found] = lookup.writer_ranks[positions[found]]
    return ranks


def __make_dataset_output_lookup(
    canonical_raw_ids: np.ndarray, nranks: int
) -> DatasetOutputLookup:
    """Build immutable output lookups from globally sorted, unique raw IDs."""
    canonical_raw_ids = np.asarray(canonical_raw_ids, dtype=np.int64)
    lengths = np.full(nranks, len(canonical_raw_ids) // nranks, dtype=np.int64)
    lengths[: len(canonical_raw_ids) % nranks] += 1
    writer_ranks = np.repeat(np.arange(nranks, dtype=np.int64), lengths)
    return DatasetOutputLookup(canonical_raw_ids, writer_ranks)


def __plan_dataset_output(
    raw_ids: np.ndarray, nranks: int
) -> tuple[np.ndarray, DatasetOutputLookup]:
    """Return canonical raw IDs and their deterministic writer assignment."""
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    canonical_raw_ids = raw_ids[np.argsort(raw_ids, kind="stable")]
    duplicate = np.flatnonzero(np.diff(canonical_raw_ids) == 0)
    if len(duplicate) > 0:
        duplicate_ids = canonical_raw_ids[duplicate]
        raise ValueError(
            "Cannot write simulation dataset with duplicate raw row IDs: "
            f"{duplicate_ids.tolist()}"
        )
    return canonical_raw_ids, __make_dataset_output_lookup(canonical_raw_ids, nranks)


def __get_dataset_output_lookup(
    raw_ids: np.ndarray, comm: MPI.Comm
) -> tuple[DatasetOutputLookup, np.ndarray]:
    """Collectively plan output ownership and return this rank's destinations."""
    gathered_raw_ids = gather_index(raw_ids, comm)
    if comm.Get_rank() == 0:
        try:
            canonical_raw_ids, lookup = __plan_dataset_output(
                gathered_raw_ids, comm.Get_size()
            )
            target_ranks = __lookup_ranks(
                lookup, __lookup_positions(lookup, gathered_raw_ids)
            )
            payload: tuple[str | None, np.ndarray | None, np.ndarray | None] = (
                None,
                canonical_raw_ids,
                target_ranks,
            )
        except ValueError as error:
            payload = (str(error), None, None)
    else:
        payload = (None, None, None)

    message, canonical_raw_ids, target_ranks = comm.bcast(payload, root=0)
    if message is not None:
        raise ValueError(message)
    assert canonical_raw_ids is not None
    assert target_ranks is not None
    lookup = __make_dataset_output_lookup(canonical_raw_ids, comm.Get_size())
    return lookup, scatter_index(target_ranks, len(raw_ids), comm)


def get_writer_rows(raw_ids: np.ndarray, comm: MPI.Comm) -> np.ndarray:
    """This rank's contiguous block of a dataset's canonical output rows."""
    lookup, _ = __get_dataset_output_lookup(raw_ids, comm)
    return lookup.raw_ids[lookup.writer_ranks == comm.Get_rank()]


def __dataset_schemas(schema: Schema) -> dict[str, Schema]:
    """Every dataset in the schema, including structure sources, by UUID."""
    datasets = {}
    for name, child in schema.children.items():
        if child.type == FileEntry.DATASET:
            datasets[str(child.children["data"].attributes["main_uuid"])] = child
        elif name != "map":
            datasets |= __dataset_schemas(child)
    return datasets


def __map_uuids(map_schema: Schema | None) -> set[str]:
    if map_schema is None:
        return set()
    uuids = {str(map_schema.attributes["reference"])}
    if (primary := map_schema.children.get("primary")) is not None:
        uuids |= set(primary.children)
    if (auxiliary := map_schema.children.get("auxiliary")) is not None:
        uuids |= {uuid for pair in auxiliary.children for uuid in pair.split("__")}
    return uuids


def get_output_lookups(
    schema: Schema, comm: MPI.Comm
) -> tuple[dict[str, Schema], dict[str, DatasetOutputLookup]]:
    """
    Local schemas and global output lookups, by UUID, of the datasets the map
    refers to. Rows must already be on their writer ranks (``get_writer_rows``).
    """
    uuids = set().union(*comm.allgather(__map_uuids(schema.children.get("map"))))
    datasets = {
        uuid: child
        for uuid, child in __dataset_schemas(schema).items()
        if uuid in uuids
    }
    lookups = {}
    for uuid, child in get_all_entries(datasets, comm):
        local_raw_ids = None if child is None else get_dataset_schema_index(child)
        __collective_error(
            f"Dataset {uuid} has no output raw row index"
            if child is not None and local_raw_ids is None
            else None,
            comm,
        )
        raw_ids = (
            np.empty(0, dtype=np.int64)
            if local_raw_ids is None
            else into_array(local_raw_ids)
        )
        lookups[uuid], _ = __get_dataset_output_lookup(raw_ids, comm)
    return datasets, lookups


def __collective_error(message: str | None, comm: MPI.Comm) -> None:
    """Raise the first local validation error on every rank."""
    messages = comm.allgather(message)
    error = next((value for value in messages if value is not None), None)
    if error is not None:
        raise ValueError(error)


def __lower_primary_values(
    source_raw_ids: np.ndarray,
    raw_targets: np.ndarray,
    reference_lookup: DatasetOutputLookup,
    target_lookup: DatasetOutputLookup,
    comm: MPI.Comm,
) -> np.ndarray:
    """Route one raw-coordinate primary slot to its source output owners."""
    source_raw_ids = np.asarray(source_raw_ids, dtype=np.int64)
    raw_targets = np.asarray(raw_targets, dtype=np.int64)
    message = None
    if len(source_raw_ids) != len(raw_targets):
        message = "Primary mapping length does not match the local reference dataset"

    source_positions = np.empty(0, dtype=np.int64)
    source_ranks = np.empty(0, dtype=np.int64)
    if message is None:
        source_positions = __lookup_positions(reference_lookup, source_raw_ids)
        missing = source_positions < 0
        if np.any(missing):
            message = (
                "Primary mapping source raw row ID is not in the output: "
                f"{int(source_raw_ids[missing][0])}"
            )
        else:
            source_ranks = __lookup_ranks(reference_lookup, source_positions)
    __collective_error(message, comm)

    output_targets = np.full(len(raw_targets), -1, dtype=np.int64)
    valid_targets = raw_targets >= 0
    output_targets[valid_targets] = __lookup_positions(
        target_lookup, raw_targets[valid_targets]
    )

    received_positions = redistribute_data(source_positions, source_ranks, comm)
    received_targets = redistribute_data(output_targets, source_ranks, comm)
    reorder = np.argsort(received_positions, kind="stable")
    received_positions = received_positions[reorder]
    received_targets = received_targets[reorder]

    expected_positions = np.flatnonzero(
        reference_lookup.writer_ranks == comm.Get_rank()
    ).astype(np.int64)
    message = None
    if not np.array_equal(received_positions, expected_positions):
        message = (
            "Primary mapping entries do not cover this rank's contiguous reference "
            "output interval"
        )
    __collective_error(message, comm)
    return received_targets


def __lower_auxiliary_values(
    raw_source: np.ndarray,
    raw_target: np.ndarray,
    source_lookup: DatasetOutputLookup,
    target_lookup: DatasetOutputLookup,
    comm: MPI.Comm,
) -> tuple[np.ndarray, np.ndarray]:
    """Route one raw-coordinate auxiliary pair to its source output owners."""
    raw_source = np.asarray(raw_source, dtype=np.int64)
    raw_target = np.asarray(raw_target, dtype=np.int64)
    __collective_error(
        None
        if len(raw_source) == len(raw_target)
        else "Auxiliary mapping source and target have different lengths",
        comm,
    )

    source_positions = __lookup_positions(source_lookup, raw_source)
    target_positions = __lookup_positions(target_lookup, raw_target)
    retained = (source_positions >= 0) & (target_positions >= 0)
    source_positions = source_positions[retained]
    target_positions = target_positions[retained]
    source_ranks = __lookup_ranks(source_lookup, source_positions)

    received_source = redistribute_data(source_positions, source_ranks, comm)
    received_target = redistribute_data(target_positions, source_ranks, comm)
    reorder = np.lexsort((received_target, received_source))
    return received_source[reorder], received_target[reorder]


def resort_simulation_collection_mpi(schema: Schema, comm: MPI.Comm):
    """Lower maps to output coordinates. Rows must already be on their writers."""
    rank_has_map = "map" in schema.children
    has_map = comm.allgather(rank_has_map)
    if not any(has_map):
        return schema

    datasets, lookups = get_output_lookups(schema, comm)

    local_map = schema.children.get("map")
    map_attributes = next(
        (
            value
            for value in comm.allgather(local_map.attributes if local_map else None)
            if value
        ),
        None,
    )
    assert map_attributes is not None
    reference_uuid = str(map_attributes["reference"])
    __collective_error(
        None
        if reference_uuid in lookups
        else "Simulation mapping reference dataset is not in the output schema",
        comm,
    )
    reference_lookup = lookups[reference_uuid]
    reference_child = datasets.get(reference_uuid)
    source_raw_ids = (
        into_array(get_dataset_schema_index(reference_child))
        if reference_child is not None
        else np.empty(0, dtype=np.int64)
    )

    local_primary = local_map.children.get("primary") if local_map is not None else None
    primary_children = {}
    for target_uuid, local_slot in get_all_entries(
        local_primary.children if local_primary is not None else {}, comm
    ):
        __collective_error(
            None
            if target_uuid in lookups
            else f"Primary mapping target dataset {target_uuid} is not in the output schema",
            comm,
        )
        target_lookup = lookups[target_uuid]
        raw_targets = (
            local_slot.columns["index"].data
            if local_slot is not None
            else np.empty(0, dtype=np.int64)
        )
        lowered = __lower_primary_values(
            source_raw_ids, raw_targets, reference_lookup, target_lookup, comm
        )

        slot_metadata = comm.allgather(
            None
            if local_slot is None
            else (
                local_slot.columns["index"].combine_strategy,
                local_slot.columns["index"].attrs,
                local_slot.attributes,
            )
        )
        strategy, attrs, attributes = next(
            value for value in slot_metadata if value is not None
        )
        writer = ColumnWriter.from_numpy_array(lowered, strategy, attrs)
        if local_slot is not None:
            primary_children[target_uuid] = local_slot._replace(
                columns={"index": writer}
            )
        elif len(lowered) > 0:
            primary_children[target_uuid] = Schema(
                target_uuid, FileEntry.COLUMNS, {}, {"index": writer}, attributes
            )

    primary_schema = Schema("primary", FileEntry.COLUMNS, primary_children, {}, {})
    local_auxiliary = (
        local_map.children.get("auxiliary") if local_map is not None else None
    )
    auxiliary_children = {}
    for pair_name, local_slot in get_all_entries(
        local_auxiliary.children if local_auxiliary is not None else {}, comm
    ):
        try:
            source_uuid, target_uuid = pair_name.split("__")
        except ValueError:
            __collective_error(
                f"Auxiliary mapping pair name is invalid: {pair_name}", comm
            )
            raise RuntimeError("unreachable")
        __collective_error(
            None
            if source_uuid in lookups
            else f"Auxiliary mapping source dataset {source_uuid} is not in the output schema",
            comm,
        )
        __collective_error(
            None
            if target_uuid in lookups
            else f"Auxiliary mapping target dataset {target_uuid} is not in the output schema",
            comm,
        )

        message = None
        if local_slot is not None:
            if "source" not in local_slot.columns or "target" not in local_slot.columns:
                message = (
                    f"Auxiliary mapping pair {pair_name} has no source or target column"
                )
            elif len(local_slot.columns["source"].data) != len(
                local_slot.columns["target"].data
            ):
                message = "Auxiliary mapping source and target have different lengths"
        __collective_error(message, comm)

        raw_source = (
            local_slot.columns["source"].data
            if local_slot is not None
            else np.empty(0, dtype=np.int64)
        )
        raw_target = (
            local_slot.columns["target"].data
            if local_slot is not None
            else np.empty(0, dtype=np.int64)
        )
        source_child = datasets.get(source_uuid)
        if source_child is not None:
            local_source_ids = into_array(get_dataset_schema_index(source_child))
            local_pairs = np.isin(raw_source, local_source_ids)
            raw_source = raw_source[local_pairs]
            raw_target = raw_target[local_pairs]
        source, target = __lower_auxiliary_values(
            raw_source, raw_target, lookups[source_uuid], lookups[target_uuid], comm
        )
        if not any(comm.allgather(len(source))):
            continue

        slot_metadata = comm.allgather(
            None
            if local_slot is None
            else (
                local_slot.columns["source"].combine_strategy,
                local_slot.columns["source"].attrs,
                local_slot.columns["target"].combine_strategy,
                local_slot.columns["target"].attrs,
                local_slot.attributes,
            )
        )
        (
            source_strategy,
            source_attrs,
            target_strategy,
            target_attrs,
            attributes,
        ) = next(value for value in slot_metadata if value is not None)
        columns = {
            "source": ColumnWriter.from_numpy_array(
                source, source_strategy, source_attrs
            ),
            "target": ColumnWriter.from_numpy_array(
                target, target_strategy, target_attrs
            ),
        }
        auxiliary_children[pair_name] = (
            local_slot._replace(columns=columns)
            if local_slot is not None
            else Schema(pair_name, FileEntry.COLUMNS, {}, columns, attributes)
        )

    auxiliary_schema = Schema(
        "auxiliary", FileEntry.COLUMNS, auxiliary_children, {}, {}
    )
    map_schema = Schema(
        "map",
        FileEntry.METADATA,
        {"primary": primary_schema, "auxiliary": auxiliary_schema},
        {},
        map_attributes,
    )
    return schema._replace(children=schema.children | {"map": map_schema})
