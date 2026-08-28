import numpy as np

from opencosmo.mpi import get_mpi

BOR_CHUNK = 1 << 28


def get_global_membership(index_to_check, index, comm):
    """For each element of local_a, is it present in the global union of b?"""
    MPI = get_mpi()

    local_a = np.ascontiguousarray(index_to_check, dtype=np.int64)
    local_b = np.ascontiguousarray(index, dtype=np.int64)

    # Size the bitmap from b alone.  Rows in `a` beyond that range simply
    # cannot be members, so we mask them off rather than widening the bitmap.
    local_max = int(local_b.max()) if local_b.size else -1
    n_rows = comm.allreduce(local_max, op=MPI.MAX) + 1

    out = np.zeros(local_a.size, dtype=bool)
    if n_rows <= 0:  # global b is empty
        return out

    bits = np.zeros((n_rows + 7) // 8, dtype=np.uint8)
    _set_bits(bits, local_b)

    # One bitwise-OR reduction and the bitmap is globally complete.
    for off in range(0, bits.size, BOR_CHUNK):
        comm.Allreduce(MPI.IN_PLACE, bits[off : off + BOR_CHUNK], op=MPI.BOR)

    in_range = (local_a >= 0) & (local_a < n_rows)
    out[in_range] = _test_bits(bits, local_a[in_range])
    # Return the matching *values* from `index_to_check`, not their positions.
    return local_a[out]


def _test_bits(bits, k):
    """Return a bool array: is bit k[i] set?"""
    return ((bits[k >> 3] >> (k & 7).astype(np.uint8)) & 1).astype(bool)


def _set_bits(bits, k):
    """Set bit k[i] in the packed uint8 array `bits`, for all i.

    Bit ordering is little-endian within each byte: row k lives at
    bit (k & 7) of byte (k >> 3).  This matches np.packbits(bitorder='little').

    The obvious implementation, np.bitwise_or.at(bits, k >> 3, 1 << (k & 7)),
    is correct but unbuffered and pathologically slow.  Instead: sort and dedupe
    the keys, then OR-reduce the runs that share a byte.  Fully vectorized.
    """
    if k.size == 0:
        return
    k = np.unique(k)  # sorted, unique
    byte_idx = (k >> 3).astype(np.intp)
    val = np.uint8(1) << (k & 7).astype(np.uint8)
    # first index of each run of equal byte_idx
    run_start = np.flatnonzero(np.r_[True, byte_idx[1:] != byte_idx[:-1]])
    bits[byte_idx[run_start]] |= np.bitwise_or.reduceat(val, run_start)
