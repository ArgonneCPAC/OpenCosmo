from mpi4py import MPI


def partition(comm: MPI.Comm, length: int) -> tuple[int, int]:
    nranks = comm.Get_size()
    rank = comm.Get_rank()
    if rank == nranks - 1:
        start = rank * (length // nranks)
        size = length - start
        return (start, size)

    start = rank * (length // nranks)
    end = (rank + 1) * (length // nranks)
    size = end - start
    return (start, size)
