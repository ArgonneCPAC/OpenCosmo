from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from uuid import UUID

import h5py
import numpy as np
from opencosmo.io.schema import FileEntry, Schema
from opencosmo.io.writer import ColumnWriter

from opencosmo.index import get_data, into_array, reindex_column

if TYPE_CHECKING:
    from opencosmo.index import DataIndex, SimpleIndex

"""
A DatasetMatchIndex is used to define the mapping between one dataset and another. Every
mapping has a source dataset and a target. The index maps rows in `source` to rows in `target`.

For large simulation suites, storing the mapping between all datasets and all other datasets (even just one way)
would be impractical. Instead, one simulation is selected as the "reference", and defines its mapping to all other
datasets in the suite. Maping between any two datasets can be determined with this information.

In cases where a row appears in two simulations but does NOT appear in the reference simulation, we include
an auxillary index. This index defines the mapping only for such items. There can only be one mapping 
between any two datasets, a primary XOR an auxillary. Mapping from sim_1 -> sim_2 can be inverted with a simple
argsort.

For example:

    sim 0 -> sim 3 (primary)
    sim 0 -> sim 4 (primary)
    sim3 -> sim 4 (auxillary)

Mapping from sim 3 -> sim 4 is:

    sim_3_4 = np.concatenate(sim_0_4[np.argsort(sim_0_3)], sim_3_4)

Note: the real implementation additionally propagates -1 for unmatched rows and
requires the inverted map to be injective (one-to-one from reference to source).

Constraints:

    The primary map must *always* be the same length as the reference dataset.
    Dataset mappings are one-to-one and use simple index arrays only.

"""

type SimpleH5pyIndex = h5py.Dataset


@dataclass(frozen=True, slots=True)
class DatasetMatchSet:
    reference_source: UUID
    primary_maps: dict[UUID, SimpleH5pyIndex]
    aux_maps: dict[tuple[UUID, UUID], tuple[SimpleH5pyIndex, SimpleH5pyIndex]]
    aliases: dict[str, UUID] = field(default_factory=dict)

    @property
    def endpoints(self) -> "frozenset[UUID]":
        """Every UUID that survived the availability filter and is actually routable.

        Contrast with ``MapLayout.endpoints``, which lists every UUID the map file
        *mentions* on disk (pre-filter).  This property lists only those that passed
        the availability check in ``read_match_set`` and are therefore present in the
        open set.

        ``reference_source`` is included even when the reference dataset was not
        opened.  That is harmless: it cannot equal any opened dataset's UUID, so it
        never falsely satisfies a coverage check.
        """
        return frozenset(
            {self.reference_source}
            | set(self.primary_maps)
            | {u for pair in self.aux_maps for u in pair}
        )

    def get_alias(self, uuid: UUID):
        for alias, ds_uuid in self.aliases.items():
            if uuid == ds_uuid:
                return alias
        return None

    def get_uuid(self, alias: str):
        return self.aliases.get(alias)

    def with_aliases(self, name_to_uuid: dict[str, UUID]):
        if diff := set(name_to_uuid.values()).difference(self.primary_maps.keys()):
            if diff != {self.reference_source}:
                raise ValueError(f"Several UUIDs are not in this map: {diff}")
        elif len(set(name_to_uuid.items())) != len(name_to_uuid):
            raise ValueError("Duplicate UUIDs detected!")

        return replace(self, aliases=self.aliases | name_to_uuid)

    def make_schema(
        self,
        new_uuids: dict[str, UUID],
        indices: dict[str, DataIndex],
        source: str | None = None,
    ) -> Schema:
        if not set(new_uuids.keys()).issubset(self.aliases.keys()):
            raise ValueError(
                "Tried to match datasets that don't appear in this mapping!"
            )
        source_alias = source or self.get_alias(self.reference_source)

        if source is not None:
            lengths = {len(into_array(index)) for index in indices.values()}
            if source not in new_uuids or len(lengths) != 1:
                raise RuntimeError("Matched datasets must have identical row counts")
            source_uuid = self.get_uuid(source)
            assert source_uuid is not None
            source_index = np.sort(into_array(indices[source]))
            new_primary = {}
            for name, uuid in new_uuids.items():
                if name == source:
                    continue
                target_uuid = self.get_uuid(name)
                assert target_uuid is not None
                mapping = get_mapping(self, source_uuid, target_uuid, source_index)
                assert mapping is not None
                new_primary[uuid] = reindex_column(indices[name], mapping)
            new_auxiliary = {}
        elif source_alias in new_uuids:
            new_primary, new_auxiliary = rebuild_single_with_source(
                self, new_uuids, indices, source_alias
            )
        else:
            source_alias = next(iter(new_uuids))
            new_primary, new_auxiliary = rebuild_single_with_new_source(
                self, new_uuids, indices, source_alias
            )

        primary_schemas: dict[str, Schema] = {}
        for new_uuid, primary_map in new_primary.items():
            writer = ColumnWriter.from_numpy_array(primary_map)
            primary_schemas[str(new_uuid)] = Schema(
                str(new_uuid),
                FileEntry.COLUMNS,
                {},
                columns={"index": writer},
                attributes={},
            )

        auxiliary_schemas: dict[str, Schema] = {}
        for (uuid_source, uuid_target), (
            index_source,
            index_target,
        ) in new_auxiliary.items():
            if len(index_source) == 0:
                continue
            source_writer = ColumnWriter.from_numpy_array(index_source)
            target_writer = ColumnWriter.from_numpy_array(index_target)
            writers = {"source": source_writer, "target": target_writer}
            name = f"{uuid_source}__{uuid_target}"

            schema = Schema(name, FileEntry.COLUMNS, {}, columns=writers, attributes={})
            auxiliary_schemas[name] = schema
        primary_schema = Schema("primary", FileEntry.COLUMNS, primary_schemas, {}, {})
        auxiliary_schema = Schema(
            "auxiliary", FileEntry.COLUMNS, auxiliary_schemas, {}, {}
        )

        children = {"primary": primary_schema, "auxiliary": auxiliary_schema}
        return Schema(
            "map",
            FileEntry.METADATA,
            children,
            {},
            {"": {"format_version": 1, "reference": new_uuids[source_alias]}},
        )


def rebuild_single_with_source(
    match_set: DatasetMatchSet,
    new_uuids: dict[str, UUID],
    indices: dict[str, DataIndex],
    source: str,
):
    """
    This is used during writing to figure out the new map. The important thing
    to appreciate about writing is data is ALWAYS written in the same order
    regardless of operations.

    This algorithm assumes mapping is one to one: Each row in the source maps to
    at most one row in the target.
    """
    new_primary_maps = {}
    old_source_uuid = match_set.get_uuid(source)
    assert old_source_uuid is not None
    source_index = into_array(indices[source])
    source_sort = np.argsort(source_index)
    for name, new_uuid in new_uuids.items():
        if name == source:
            continue
        old_target_uuid = match_set.get_uuid(name)
        assert old_target_uuid is not None
        primary_map = get_primary_mapping(
            match_set, old_source_uuid, old_target_uuid, source_index
        )
        target_index = into_array(indices[name])
        target_sort = np.argsort(target_index)

        new_primary_maps[new_uuid] = reindex_column(
            target_index[target_sort], primary_map[source_sort]
        )

    new_auxiliary_maps = {}
    for (uuida, uuidb), (
        aux_source_index,
        aux_target_index,
    ) in match_set.aux_maps.items():
        aliasa = match_set.get_alias(uuida)
        aliasb = match_set.get_alias(uuidb)
        if aliasa not in new_uuids or aliasb not in new_uuids:
            continue
        indexa = into_array(indices[aliasa])
        indexb = into_array(indices[aliasb])
        reindexa = reindex_column(np.sort(indexa), aux_source_index[:])
        reindexb = reindex_column(np.sort(indexb), aux_target_index[:])
        to_keep = (reindexa != -1) & (reindexb != -1)
        if to_keep.any():
            new_auxiliary_maps[(new_uuids[aliasa], new_uuids[aliasb])] = (
                reindexa[to_keep],
                reindexb[to_keep],
            )

    return new_primary_maps, new_auxiliary_maps


def rebuild_single_with_new_source(
    match_set: DatasetMatchSet,
    new_uuids: dict[str, UUID],
    indices: dict[str, DataIndex],
    source: str,
):
    new_primary_maps = {}
    old_source_uuid = match_set.get_uuid(source)
    assert old_source_uuid is not None

    source_index = into_array(indices[source])
    source_sort = np.argsort(source_index)

    for target, new_uuid in new_uuids.items():
        if target == source:
            continue
        old_target_uuid = match_set.get_uuid(target)
        assert old_target_uuid is not None
        mapping = get_mapping(match_set, old_source_uuid, old_target_uuid, source_index)
        assert mapping is not None
        target_index = into_array(indices[target])
        new_primary_maps[new_uuid] = reindex_column(
            np.sort(target_index), mapping[source_sort]
        )

    new_auxiliary_maps = {}
    aliases = sorted(alias for alias in new_uuids if alias != source)
    for position, aliasa in enumerate(aliases):
        old_uuida = match_set.get_uuid(aliasa)
        assert old_uuida is not None
        indexa = into_array(indices[aliasa])
        sorteda = np.sort(indexa)
        for aliasb in aliases[position + 1 :]:
            old_uuidb = match_set.get_uuid(aliasb)
            assert old_uuidb is not None
            indexb = into_array(indices[aliasb])
            mapping = get_mapping(match_set, old_uuida, old_uuidb, indexa)
            assert mapping is not None
            sort_a = np.argsort(indexa)
            reindexa = reindex_column(sorteda, indexa[sort_a])
            reindexb = reindex_column(np.sort(indexb), mapping[sort_a])
            keep = (reindexa >= 0) & (reindexb >= 0)

            primary_a = new_primary_maps[new_uuids[aliasa]]
            primary_b = new_primary_maps[new_uuids[aliasb]]
            routed_b_by_a = np.full(len(indexa), -1, dtype=np.int64)
            routed = (primary_a >= 0) & (primary_b >= 0)
            routed_b_by_a[primary_a[routed]] = primary_b[routed]

            routed_pair = np.zeros(len(reindexa), dtype=bool)
            valid_a = reindexa >= 0
            routed_pair[valid_a] = routed_b_by_a[reindexa[valid_a]] == reindexb[valid_a]
            keep &= ~routed_pair
            if keep.any():
                new_auxiliary_maps[(new_uuids[aliasa], new_uuids[aliasb])] = (
                    reindexa[keep],
                    reindexb[keep],
                )

    return new_primary_maps, new_auxiliary_maps


def get_mapping(
    match_set: DatasetMatchSet, source: UUID | str, target: UUID | str, index: DataIndex
) -> SimpleIndex | None:
    source_uuid = match_set.aliases.get(str(source), source)
    target_uuid = match_set.aliases.get(str(target), target)

    if not isinstance(source_uuid, UUID) or not isinstance(target_uuid, UUID):
        raise ValueError("Mapping names must be UUIDs or registered aliases")

    try:
        auxiliary_mapping = get_auxillary_mapping(
            match_set, source_uuid, target_uuid, index
        )
        mapping = get_primary_mapping(match_set, source_uuid, target_uuid, index)
    except KeyError as exc:
        raise ValueError(
            f"Unable to map from '{source}' to '{target}': no primary mapping route "
            "exists between these datasets."
        ) from exc
    if auxiliary_mapping is None:
        return mapping

    aux_index, aux_mapping = auxiliary_mapping
    mapping[aux_index] = aux_mapping
    return mapping


def get_auxillary_mapping(
    match_set: DatasetMatchSet, source: UUID, target: UUID, index: DataIndex
):
    auxillary_map = match_set.aux_maps.get((source, target))
    if auxillary_map is None:
        auxillary_map = match_set.aux_maps.get((target, source))
        if auxillary_map is None:
            return None
        auxillary_map = (auxillary_map[1], auxillary_map[0])

    auxillary_map = (auxillary_map[0][:], auxillary_map[1][:])

    index_arr = into_array(index)

    # np.intersect1d would report only the first position of each repeated
    # value, silently leaving later duplicates without their override. An index
    # can contain duplicates after a match, because several source rows may map
    # onto the same target row.
    aux_source, aux_target = auxillary_map
    order = np.argsort(aux_source, kind="stable")
    sorted_source = aux_source[order]
    candidates = np.searchsorted(sorted_source, index_arr)
    candidates[candidates >= len(sorted_source)] = 0
    index_into_final = np.flatnonzero(sorted_source[candidates] == index_arr)
    index_into_map = order[candidates[index_into_final]]
    return (index_into_final, aux_target[index_into_map])


def get_primary_mapping(
    match_set: DatasetMatchSet, source: UUID, target: UUID, index: DataIndex
):
    if source == match_set.reference_source:
        mapping = match_set.primary_maps[target]
        return get_data(mapping, index)

    elif target == match_set.reference_source:
        mapping = match_set.primary_maps[source]
        return __get_inverse_mapping(mapping, index, match_set.reference_source, source)

    map_to_source = match_set.primary_maps[source]
    map_to_target = match_set.primary_maps[target]

    ref_rows = __get_inverse_mapping(
        map_to_source, index, match_set.reference_source, source
    )
    result = np.full(len(ref_rows), -1, dtype=np.int64)
    valid = ref_rows != -1
    if valid.any():
        forward_target = map_to_target[:]
        result[valid] = forward_target[ref_rows[valid]]
    return result


def __get_inverse_mapping(
    mapping: SimpleH5pyIndex,
    index: "DataIndex",
    reference_name: "str | UUID",
    source_name: "str | UUID",
) -> np.ndarray:
    """Invert a reference→source primary map for the given source rows.

    Parameters
    ----------
    mapping:
        An h5py Dataset of length ``n_reference`` whose value at reference row
        ``r`` is the absolute source row that ``r`` maps to, or ``-1`` if
        unmatched.  This is the *primary map* stored on disk.
    index:
        Absolute row numbers of the *source* dataset currently selected.
        These are source-coordinate values, NOT reference-coordinate values.
    reference_name:
        Human-readable name of the reference simulation (used in error messages).
    source_name:
        Human-readable name of the source simulation (used in error messages).

    Returns
    -------
    result : np.ndarray, dtype int64, shape (len(into_array(index)),)
        ``result[i]`` is the absolute reference row corresponding to
        ``into_array(index)[i]``, or ``-1`` if no reference row maps to it.

    Raises
    ------
    ValueError
        If the mapping is one-to-many (not injective), meaning inversion is
        ambiguous.  The check is performed over the *entire* map (not just the
        rows in ``index``) so the error is deterministic regardless of prior
        filtering.
    """
    # Inversion is inherently non-lazy: we must read the entire forward map to
    # know which reference row points at a given source row.  This is bounded
    # by the reference dataset length, which is acceptable.
    forward = mapping[:]

    ref_rows = np.flatnonzero(forward != -1)
    source_rows = forward[ref_rows]

    order = np.argsort(source_rows, kind="stable")
    sorted_source = source_rows[order]

    # Injectivity check: performed globally over the whole map so the error is
    # deterministic regardless of which rows happen to be in `index`.
    if len(sorted_source) > 1:
        n = int(np.count_nonzero(sorted_source[1:] == sorted_source[:-1]))
        if n > 0:
            raise ValueError(
                f"Cannot match with source='{source_name}': the mapping from "
                f"'{reference_name}' to '{source_name}' is one-to-many "
                f"({n} ambiguous rows), so it cannot be inverted. "
                f"Match with source='{reference_name}' instead."
            )

    wanted = into_array(index)
    result = np.full(len(wanted), -1, dtype=np.int64)

    if len(sorted_source) == 0:
        return result

    pos = np.searchsorted(sorted_source, wanted)
    pos = np.clip(pos, 0, len(sorted_source) - 1)

    # searchsorted returns an insertion point for absent values, not an error.
    # We must verify that the located position actually holds the wanted value
    # before writing; otherwise absent source rows would silently alias to a
    # neighbouring entry.
    hit = sorted_source[pos] == wanted
    result[hit] = ref_rows[order[pos[hit]]]

    return result
