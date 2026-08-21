from functools import cache
from typing import Optional

import numpy as np
from mpi4py.util import dtlib

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None  # type: ignore


def has_mpi() -> bool:
    return get_comm_world() is not None


@cache
def get_comm_world() -> Optional["MPI.Comm"]:
    if MPI is None or MPI.COMM_WORLD.Get_size() == 1:
        return None
    return MPI.COMM_WORLD.Dup()


def get_mpi():
    return MPI


def parallel_assert(condition: bool, comm: MPI.Comm | None = None):
    comm = comm or get_comm_world()
    assert comm is not None
    is_bool = comm.allgather(isinstance(condition, (np.bool, bool)))
    if not all(is_bool):
        raise ValueError("Expected a boolean condition on all ranks")
    passes = comm.allgather(condition)
    failed = [i for i in range(len(passes)) if not passes[i]]
    if failed:
        raise ValueError(f"Parallel assertion failed on ranks {failed}")


def parallel_assert_is_simple_index(value, comm: MPI.Comm | None = None):
    parallel_assert(isinstance(value, np.ndarray), comm)
    parallel_assert(value.ndim == 1, comm)
    parallel_assert(value.dtype == np.int64, comm)


def parallel_assert_same_dtype(value: np.ndarray, comm: MPI.Comm | None = None):
    parallel_assert(isinstance(value, np.ndarray), comm)
    assert comm is not None
    all_dtypes = comm.allgather(value.dtype)
    parallel_assert(len(set(all_dtypes)) == 1, comm)


def parallel_assert_compatible_shapes(value: np.ndarray, comm: MPI.Comm | None = None):
    parallel_assert(isinstance(value, np.ndarray), comm)
    assert comm is not None
    shapes = [s[1:] for s in comm.allgather(value.shape)]
    parallel_assert(len(set(shapes)) == 1, comm)


def parallel_assert_can_stack(value: np.ndarray, comm: MPI.Comm | None = None):
    parallel_assert_same_dtype(value, comm)
    parallel_assert_compatible_shapes(value, comm)


def get_subcom(include: list[bool], comm: MPI.Comm):
    group = comm.Get_group()
    new_group = group.Incl(np.where(include)[0])
    new_comm = comm.Create(new_group)
    group.free()
    return new_comm, new_group


def gather_index(index: np.ndarray, comm: MPI.Comm):
    parallel_assert_is_simple_index(index, comm)

    counts = comm.gather(len(index))
    if comm.Get_rank() == 0:
        displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]

        total_size = sum(counts)
        recvbuf = np.empty(total_size, dtype=np.int64)
    else:
        counts = None
        displacements = None
        recvbuf = None

        # 4. Perform the variable-length gather operation
        # Note the uppercase 'G' which indicates a buffer/array optimization wrapper
    comm.Gatherv(
        sendbuf=index, recvbuf=[recvbuf, counts, displacements, MPI.INT64_T], root=0
    )
    return recvbuf


def scatter_index(index: np.ndarray | None, length: int, comm: MPI.Comm):
    counts = comm.allgather(length)
    index_length = comm.bcast(len(index) if index is not None else None)
    parallel_assert(np.sum(counts) == index_length, comm)

    if comm.Get_rank() == 0:
        displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]

    else:
        counts = None
        displacements = None

    recvbuf = np.empty(length, np.int64)
    comm.Scatterv(
        [index, counts, displacements, MPI.INT64_T],
        recvbuf,
        root=0,
    )
    return recvbuf
    # 4. Perform the variable-length gather operation
    # Note the uppercase 'G' which indicates a buffer/array optimization wrapper


def redistribute_data(data: np.ndarray, target_rank: np.ndarray, comm: MPI.Comm):
    parallel_assert_is_simple_index(target_rank, comm)
    parallel_assert(len(target_rank) == len(data), comm)
    parallel_assert_can_stack(data, comm)

    rank_distribution = [np.where(target_rank == i)[0] for i in range(comm.Get_size())]
    lengths = np.array([len(tr) for tr in rank_distribution]).astype(np.int64)
    parallel_assert(sum(lengths) == len(data), comm)
    recv_lengths = np.empty(comm.Get_size(), dtype=np.int64)
    comm.Alltoall(lengths, recv_lengths)

    send_displs = np.zeros(comm.Get_size(), dtype="i")
    send_displs[1:] = np.cumsum(lengths)[:-1]

    recv_displs = np.zeros(comm.Get_size(), dtype="i")
    recv_displs[1:] = np.cumsum(recv_lengths)[:-1]

    # --- Build the actual send buffer ---
    # Total data this rank sends = sum(send_counts)
    sendbuf = np.concat([data[rank_distribution[i]] for i in range(comm.Get_size())])

    # --- Allocate the receive buffer ---
    total_recv = int(np.sum(recv_lengths))
    recvbuf = np.empty(total_recv, dtype=data.dtype)

    # --- The actual call ---
    dtype = dtlib.from_numpy_dtype(data.dtype)
    comm.Alltoallv(
        [sendbuf, lengths, send_displs, dtype],
        [recvbuf, recv_lengths, recv_displs, dtype],
    )
    return recvbuf
