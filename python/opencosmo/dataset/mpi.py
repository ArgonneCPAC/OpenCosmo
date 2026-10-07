from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from warnings import warn

from opencosmo.index.build import single_chunk
from opencosmo.plugins.contexts import HookPoint, PartitionCtx
from opencosmo.plugins.hook import query
from opencosmo.spatial.protocols import TreePartition

if TYPE_CHECKING:
    import h5py
    from mpi4py import MPI

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.spatial.protocols import SpatialIndex


def partition(
    comm: MPI.Comm,
    header: OpenCosmoHeader,
    index_group: h5py.Group,
    data_group: h5py.Group,
    spatial_index: SpatialIndex | None,
    min_level: Optional[int] = None,
) -> Optional[TreePartition]:
    """
    When opening with MPI, each rank recieves an equally-sized chunk of the
    spatial index. In principle this means the number of objects are similar
    between ranks.
    """
    partition_plugin_result = query(
        HookPoint.Partition,
        PartitionCtx(
            comm,
            header,
            index_group,
            data_group,
            spatial_index,
            min_level,
        ),
    )
    if partition_plugin_result is not None:
        return partition_plugin_result

    if spatial_index is not None:
        from opencosmo.spatial.index import partition as partition_spatial_index

        partitions = partition_spatial_index(
            spatial_index,
            comm.Get_size(),
            index_group,
            min_level,
        )
        try:
            part = partitions[comm.Get_rank()]
        except IndexError:
            warn(
                "This MPI Rank recieved no data. "
                "The tree doesn't have enough subdivisions to serve every rank!"
            )
            part = None

        return part

    length = len(next(iter(data_group.values())))
    nranks = comm.Get_size()
    rank = comm.Get_rank()
    if rank == nranks - 1:
        start = rank * (length // nranks)
        size = length - start
        index = single_chunk(start, size)

    else:
        start = rank * (length // nranks)
        end = (rank + 1) * (length // nranks)
        size = end - start

        index = single_chunk(start, size)

    return TreePartition(index, None, None)
