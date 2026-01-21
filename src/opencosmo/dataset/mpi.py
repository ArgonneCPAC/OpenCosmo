from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from warnings import warn

from opencosmo.index.build import single_chunk
from opencosmo.spatial.protocols import TreePartition

if TYPE_CHECKING:
    import h5py
    from mpi4py import MPI

    from opencosmo.spatial.tree import Tree


def partition(
    comm: MPI.Comm, length: int, counts: h5py.Group, tree: Optional[Tree]
) -> Optional[TreePartition]:
    """
    When opening with MPI, each rank recieves an equally-sized chunk of the
    spatial index. In principle this means the number of objects are similar
    between ranks.
    """
    if tree is not None:
        partitions = tree.partition(comm.Get_size(), counts)
        try:
            part = partitions[comm.Get_rank()]
        except IndexError:
            warn(
                "This MPI Rank recieved no data. "
                "The tree doesn't have enough subdivisions to serve every rank!"
            )
            part = None
        return part

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
