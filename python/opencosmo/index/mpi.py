from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from opencosmo.mpi import get_mpi

if TYPE_CHECKING:
    from mpi4py import MPI

BOR_CHUNK = 1 << 28


def is_in_global_index(
    values: npt.ArrayLike, index: npt.ArrayLike, comm: MPI.Comm
) -> npt.NDArray[np.bool_]:
    """
    Test each local value for membership in the union of `index` across all ranks.

    Collective: every rank must call this with the same call sequence, because
    the global row count and the bitmap reduction are both collective.

    Parameters
    ----------
    values : array_like
        Local candidate row values to test. Not required to be sorted or unique.
    index : array_like
        This rank's contribution to the global index. The union over all ranks
        defines membership.
    comm : MPI.Comm
        The communicator to reduce over.

    Returns
    -------
    numpy.ndarray
        Boolean mask aligned with `values`, true where the value is present in
        the global index.
    """
    mpi = get_mpi()

    local_values = np.ascontiguousarray(values, dtype=np.int64)
    local_index = np.ascontiguousarray(index, dtype=np.int64)

    # Size the bitmap from the index alone. Values beyond that range simply
    # cannot be members, so we mask them off rather than widening the bitmap.
    local_max = int(local_index.max()) if local_index.size else -1
    n_rows = comm.allreduce(local_max, op=mpi.MAX) + 1

    result = np.zeros(local_values.size, dtype=bool)
    if n_rows <= 0:  # the global index is empty, so nothing can be a member
        return result

    bits = np.zeros((n_rows + 7) // 8, dtype=np.uint8)
    __set_bits(bits, local_index)

    # One bitwise-OR reduction and the bitmap is globally complete.
    for offset in range(0, bits.size, BOR_CHUNK):
        comm.Allreduce(mpi.IN_PLACE, bits[offset : offset + BOR_CHUNK], op=mpi.BOR)

    in_range = (local_values >= 0) & (local_values < n_rows)
    result[in_range] = __test_bits(bits, local_values[in_range])
    return result


def __test_bits(
    bits: npt.NDArray[np.uint8], keys: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """Return a bool array: is bit keys[i] set?"""
    return ((bits[keys >> 3] >> (keys & 7).astype(np.uint8)) & 1).astype(bool)


def __set_bits(bits: npt.NDArray[np.uint8], keys: npt.NDArray[np.int64]) -> None:
    """Set bit keys[i] in the packed uint8 array `bits`, for all i.

    Bit ordering is little-endian within each byte: row k lives at
    bit (k & 7) of byte (k >> 3). This matches np.packbits(bitorder='little').

    The obvious implementation, np.bitwise_or.at(bits, k >> 3, 1 << (k & 7)),
    is correct but unbuffered and pathologically slow. Instead: sort and dedupe
    the keys, then OR-reduce the runs that share a byte. Fully vectorized.
    """
    if keys.size == 0:
        return
    keys = np.unique(keys)  # sorted, unique
    byte_idx = (keys >> 3).astype(np.intp)
    val = np.uint8(1) << (keys & 7).astype(np.uint8)
    # first index of each run of equal byte_idx
    run_start = np.flatnonzero(np.r_[True, byte_idx[1:] != byte_idx[:-1]])
    bits[byte_idx[run_start]] |= np.bitwise_or.reduceat(val, run_start)
