from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np

from opencosmo.index import (
    get_data,
    into_array,
)

if TYPE_CHECKING:
    from uuid import UUID

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

    sim_3_4 = np.concatenate(np.argsort(sim_0_3)[sim_0_4], sim_3_4)


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


def get_mapping(
    match_set: DatasetMatchSet, source: UUID, target: UUID, index: DataIndex
) -> DataIndex | None:
    auxilliary_mapping = get_auxillary_mapping(match_set, source, target, index)
    mapping = get_primary_mapping(match_set, source, target, index)
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
        return __get_inverse_mapping(mapping, index)
        # todo handle missing matches
    map_to_source = match_set.primary_maps[source]
    map_to_target = get_data(match_set.primary_maps[target], index)
    assert isinstance(map_to_source, h5py.Dataset)

    map_from_source = __get_inverse_mapping(map_to_source, index)
    result = np.full(len(map_to_target), -1, dtype=np.int64)
    matched = map_to_target != -1
    result[matched] = map_from_source[map_to_target[matched]]
    return result


def __get_inverse_mapping(mapping: SimpleH5pyIndex, index: DataIndex):
    mapping_index = get_data(mapping, index)

    valid_sources = np.where(mapping_index != -1)[0]
    if len(valid_sources) == 0:
        return mapping_index

    valid_targets = mapping_index[valid_sources]
    n_target = valid_targets.max() + 1
    final_map = np.full(n_target, -1, dtype=np.int64)
    final_map[valid_targets] = valid_sources
    return final_map
