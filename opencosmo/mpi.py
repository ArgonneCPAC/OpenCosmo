from functools import cache
from typing import Optional

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None  # type: ignore


@cache
def get_comm_world() -> Optional["MPI.Comm"]:
    if MPI is None or MPI.COMM_WORLD.Get_size() == 1:
        return None
    return MPI.COMM_WORLD.Dup()


def get_mpi():
    return MPI
