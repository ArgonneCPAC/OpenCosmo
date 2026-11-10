from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import h5py
import numpy as np

from opencosmo.index.build import empty
from opencosmo.plugins.contexts import HookPoint
from opencosmo.plugins.hook import hook
from opencosmo.spatial.index import partition
from opencosmo.spatial.protocols import TreePartition

if TYPE_CHECKING:
    from opencosmo.plugins.contexts import PartitionCtx


@hook(
    HookPoint.Partition,
    when=lambda ctx: (
        ctx.header.file.data_type == "snapshot_particles"
        and ctx.spatial_index is not None
    ),
)
def _partition_all_types_alike(ctx: PartitionCtx) -> TreePartition:
    """Give every particle type of a snapshot the same region on each rank.

    The octree cells are split between ranks using the counts of all types
    added together, so the split does not depend on which type is opened.
    """
    assert ctx.spatial_index is not None
    counts = __combined_counts(ctx.index_group.parent.parent)
    partitions = partition(
        ctx.spatial_index, ctx.comm.Get_size(), counts, ctx.min_level
    )
    rank = ctx.comm.Get_rank()
    if rank >= len(partitions):
        warn(
            "This MPI Rank recieved no data. "
            "The tree doesn't have enough subdivisions to serve every rank!"
        )
        return TreePartition(empty(), None, None)
    return partitions[rank]


def __combined_counts(container: h5py.Group) -> dict[str, dict[str, np.ndarray]]:
    """The number of particles of all types in each octree cell, for the levels
    that every type's index has."""
    indices = [
        group["index"]
        for group in container.values()
        if isinstance(group, h5py.Group) and "index" in group and "data" in group
    ]
    levels = set.intersection(*({key for key in index} for index in indices))
    return {
        level: {"size": np.sum([index[level]["size"][:] for index in indices], axis=0)}
        for level in sorted(levels)
    }
