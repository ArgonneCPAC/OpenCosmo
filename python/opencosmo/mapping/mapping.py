from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from uuid import UUID

import h5py
import numpy as np

from opencosmo.index import get_data, into_array

if TYPE_CHECKING:
    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex

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

    The primary map must *always* be the same length as the primary dataset. 
    Mapping from one -> many is allowed, but disables inversion functionalities.

"""

type SimpleH5pyIndex = h5py.Dataset
type ChunkedH5pyIndex = tuple[h5py.Dataset, h5py.Dataset]

H5pyIndex = SimpleH5pyIndex | ChunkedH5pyIndex


@dataclass(frozen=True, slots=True)
class DatasetMatchSet:
    reference_source: UUID
    primary_maps: dict[UUID, H5pyIndex]
    aux_maps: dict[tuple[UUID, UUID], tuple[H5pyIndex, H5pyIndex]]
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

    def with_aliases(self, name_to_uuid: dict[str, UUID]):
        if diff := set(name_to_uuid.values()).difference(self.primary_maps.keys()):
            if diff != {self.reference_source}:
                raise ValueError(f"Several UUIDs are not in this map: {diff}")

        return replace(self, aliases=self.aliases | name_to_uuid)


def get_mapping(
    match_set: DatasetMatchSet, source: UUID | str, target: UUID | str, index: DataIndex
) -> DataIndex | None:
    source_uuid = match_set.aliases.get(str(source), source)
    target_uuid = match_set.aliases.get(str(target), target)

    if not isinstance(source_uuid, UUID) or not isinstance(target_uuid, UUID):
        raise ValueError("Mapping names must be UUIDs or registered aliases")

    auxilliary_mapping = get_auxillary_mapping(
        match_set, source_uuid, target_uuid, index
    )
    mapping = get_primary_mapping(match_set, source_uuid, target_uuid, index)
    if auxilliary_mapping is None:
        return mapping

    if isinstance(mapping, tuple):
        assert isinstance(auxilliary_mapping[1], (tuple))
        return __build_chunked_mapping(mapping, auxilliary_mapping)

    aux_index, aux_mapping = auxilliary_mapping
    mapping[aux_index] = aux_mapping
    return mapping


def __build_chunked_mapping(
    primary_mapping: ChunkedIndex, auxilliary_mapping: tuple[SimpleIndex, ChunkedIndex]
):
    starts, sizes = primary_mapping

    aux_index, aux_mapping = auxilliary_mapping
    starts[aux_index] = aux_mapping[0]
    sizes[aux_index] = aux_mapping[1]
    return (starts, sizes)


def get_auxillary_mapping(
    match_set: DatasetMatchSet, source: UUID, target: UUID, index: DataIndex
):
    auxillary_map = match_set.aux_maps.get((source, target))
    if auxillary_map is None:
        auxillary_map = match_set.aux_maps.get((target, source))
        if auxillary_map is None:
            return None
        auxillary_map = (auxillary_map[1], auxillary_map[0])

    assert isinstance(auxillary_map[0], h5py.Dataset)

    def make_arrays(aux_map):
        aux_index = aux_map[0][:]
        if isinstance(aux_map[1], tuple):
            return (aux_index, make_arrays(aux_map[1]))
        return (aux_index, aux_map[1][:])

    auxillary_map = make_arrays(auxillary_map)

    index_arr = into_array(index)
    _, index_into_map, index_into_final = np.intersect1d(
        auxillary_map[0], index_arr, return_indices=True
    )
    if isinstance(auxillary_map[1], tuple):
        return (
            index_into_final,
            (
                auxillary_map[1][0][index_into_map],
                auxillary_map[1][1][index_into_map],
            ),
        )

    return (index_into_final, auxillary_map[1][index_into_map])


def get_primary_mapping(
    match_set: DatasetMatchSet, source: UUID, target: UUID, index: DataIndex
):
    if source == match_set.reference_source:
        mapping = match_set.primary_maps[target]
        if isinstance(mapping, tuple):
            return (get_data(mapping[0], index), get_data(mapping[1], index))
        return get_data(mapping, index)

    elif target == match_set.reference_source:
        mapping = match_set.primary_maps[source]
        assert not isinstance(mapping, tuple)
        return __get_inverse_mapping(mapping, index, match_set.reference_source, source)

    map_to_source = match_set.primary_maps[source]
    map_to_target = match_set.primary_maps[target]
    assert isinstance(map_to_source, h5py.Dataset)
    assert isinstance(map_to_target, h5py.Dataset)

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
