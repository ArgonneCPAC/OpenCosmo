from typing import Optional
from warnings import warn

from mpi4py import MPI

from opencosmo.dataset.index import ChunkedIndex
from opencosmo.spatial.tree import Tree


def partition(comm: MPI.Comm, length: int, tree: Optional[Tree]) -> ChunkedIndex:
    if tree is not None:
        partitions = tree.partition(comm.Get_size())
        try:
            part = partitions[comm.Get_rank()]
        except IndexError:
            warn(
                "This MPI Rank recieved no data. "
                "The tree doesn't have enough subdivisions to serve every rank!"
            )
            part = ChunkedIndex.empty()
        return part

    nranks = comm.Get_size()
    rank = comm.Get_rank()
    if rank == nranks - 1:
        start = rank * (length // nranks)
        size = length - start
        return ChunkedIndex.single_chunk(start, size)

    start = rank * (length // nranks)
    end = (rank + 1) * (length // nranks)
    size = end - start

    return ChunkedIndex.single_chunk(start, size)
