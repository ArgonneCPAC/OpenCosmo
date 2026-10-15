"""
Direct reads of Blosc2-compressed HDF5 columns.

HDF5 issues one read per chunk per column. When a column's filters are Blosc2
(optionally followed by fletcher32), its chunks can instead be fetched straight
from the file, with neighbouring chunks of every column merged into one
positional read, and decoded with ``blosc2``. Eligibility and each chunk's byte
range are worked out once per column as a ``ChunkMap``; columns without one are
read through HDF5.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from .read_plan import plan, scatter
from .unary import get_length

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    import h5py

    from opencosmo.index import DataIndex, IndexArray

    from .read_plan import ReadPlan

BLOSC2_FILTER = 32026
FLETCHER32_FILTER = 3
# Chunks this close together on disk are fetched in one read, gap included.
MERGE_GAP = 1 << 20
# Row length for summing fletcher32 words; any row sum fits in uint32.
FLETCHER_BLOCK = 1024

type LastChunks = MutableMapping[str, tuple[int, np.ndarray]]


@dataclass(frozen=True, eq=False)
class ChunkMap:
    """
    Where each storage chunk of one column lives in its file. Chunk ``k`` holds
    rows ``[k * chunk_rows, (k + 1) * chunk_rows)`` and is stored in
    ``[offsets[k], offsets[k] + sizes[k])``, including the fletcher32 trailer if
    ``fletcher32``.
    """

    path: str
    dtype: np.dtype
    row_shape: tuple[int, ...]
    n_rows: int
    chunk_rows: int
    offsets: IndexArray
    sizes: IndexArray
    fletcher32: bool


def chunk_map(ds: h5py.Dataset) -> ChunkMap | None:
    """
    Map the chunks of ``ds``, or return ``None`` if it must be read through HDF5:
    it is not chunked along rows only, its filters are not Blosc2 (plus
    fletcher32), its file is not a plain file on disk, or a chunk is missing or
    was stored unfiltered.
    """
    file = ds.file
    if (
        ds.chunks is None
        or ds.chunks[1:] != ds.shape[1:]
        or ds.dtype.hasobject
        or file.driver != "sec2"
        or file.userblock_size != 0
        or not os.path.isfile(file.filename)
    ):
        return None
    create = ds.id.get_create_plist()
    filters = tuple(create.get_filter(i)[0] for i in range(create.get_nfilters()))
    if filters not in ((BLOSC2_FILTER,), (BLOSC2_FILTER, FLETCHER32_FILTER)):
        return None

    chunk_rows = ds.chunks[0]
    n_chunks = -(-ds.shape[0] // chunk_rows)
    stored: list[tuple[int, int, int, int]] = []
    ds.id.chunk_iter(
        lambda info: stored.append(
            (info.chunk_offset[0], info.byte_offset, info.size, info.filter_mask)
        )
    )
    if n_chunks == 0 or len(stored) != n_chunks:
        return None
    rows, offsets, sizes, masks = np.array(sorted(stored), dtype=np.int64).T
    if np.any(masks != 0) or np.any(rows != np.arange(n_chunks) * chunk_rows):
        return None
    return ChunkMap(
        file.filename,
        ds.dtype,
        ds.shape[1:],
        ds.shape[0],
        chunk_rows,
        offsets,
        sizes,
        len(filters) == 2,
    )


def read_direct(
    columns: Mapping[str, ChunkMap],
    index: DataIndex,
    budget: int,
    last_chunks: LastChunks | None = None,
) -> dict[str, np.ndarray]:
    """
    Read ``index`` from mapped columns. Chunks of every column are fetched in
    file order, neighbours merged into reads of at most ``budget`` bytes, and
    each chunk is decoded once. ``last_chunks`` keeps each column's most
    recently decoded chunk across calls, so successive small reads from one
    chunk decode it once.
    """
    paths = {cmap.path for cmap in columns.values()}
    if len(paths) > 1:
        raise ValueError("Columns read together must come from one file")
    if get_length(index) == 0:
        return {name: np.array([], cmap.dtype) for name, cmap in columns.items()}
    import blosc2

    # One window per touched chunk: merging is done on byte ranges instead.
    plans: dict[tuple[int, int, int], ReadPlan] = {}
    for cmap in columns.values():
        layout = (cmap.n_rows, cmap.chunk_rows, __row_bytes(cmap))
        if layout not in plans:
            plans[layout] = plan(index, *layout, layout[1] * layout[2])

    outputs: dict[str, np.ndarray] = {}
    column_plans: dict[str, ReadPlan] = {}
    reads: list[tuple[int, int, str, int]] = []
    for name, cmap in columns.items():
        read_plan = plans[(cmap.n_rows, cmap.chunk_rows, __row_bytes(cmap))]
        column_plans[name] = read_plan
        outputs[name] = np.empty((read_plan.length,) + cmap.row_shape, cmap.dtype)
        cached = last_chunks.get(name) if last_chunks is not None else None
        for w, chunk_id in enumerate(read_plan.windows.row_lo // cmap.chunk_rows):
            if cached is not None and cached[0] == chunk_id:
                __place(read_plan, w, cmap, cached[1], outputs[name])
                continue
            reads.append(
                (int(cmap.offsets[chunk_id]), int(cmap.sizes[chunk_id]), name, w)
            )
    # File order; the name breaks ties so the order is deterministic.
    reads.sort()

    previous_threads = blosc2.set_nthreads(__blosc_threads())
    fd = os.open(paths.pop(), os.O_RDONLY)
    try:
        for group in __merge(reads, budget):
            lo = group[0][0]
            size = max(offset + n for offset, n, _, _ in group) - lo
            raw = os.pread(fd, size, lo)
            if len(raw) != size:
                raise OSError(f"Short read at byte {lo} of {columns[group[0][2]].path}")
            buffer = memoryview(raw)
            for offset, n, name, w in group:
                cmap = columns[name]
                read_plan = column_plans[name]
                chunk = __decode(cmap, buffer[offset - lo : offset - lo + n])
                __place(read_plan, w, cmap, chunk, outputs[name])
                if last_chunks is not None:
                    k = int(read_plan.windows.row_lo[w]) // cmap.chunk_rows
                    last_chunks[name] = (k, chunk)
    finally:
        os.close(fd)
        blosc2.set_nthreads(previous_threads)

    return {
        name: out
        if column_plans[name].gather is None
        else out[column_plans[name].gather]
        for name, out in outputs.items()
    }


@cache
def __blosc_threads() -> int:
    """
    Blosc decodes with its own threads. Under MPI the ranks already share the
    node's cores, so each decodes on one.
    """
    from opencosmo.mpi import get_mpi

    mpi = get_mpi()
    if mpi is not None and mpi.COMM_WORLD.Get_size() > 1:
        return 1
    return min(os.cpu_count() or 1, 8)


def __row_bytes(cmap: ChunkMap) -> int:
    return max(1, cmap.dtype.itemsize * int(np.prod(cmap.row_shape)))


def __merge(
    reads: list[tuple[int, int, str, int]], budget: int
) -> list[list[tuple[int, int, str, int]]]:
    """
    Group chunk reads sorted by offset into ranges that are read at once.
    Neighbours join a range when the gap is at most ``MERGE_GAP`` and the range
    stays within ``budget``; a chunk larger than the budget is read alone.
    """
    groups: list[list[tuple[int, int, str, int]]] = []
    lo = hi = 0
    for read in reads:
        offset, size, _, _ = read
        if (
            groups
            and offset - hi <= MERGE_GAP
            and max(hi, offset + size) - lo <= budget
        ):
            groups[-1].append(read)
            hi = max(hi, offset + size)
            continue
        groups.append([read])
        lo, hi = offset, offset + size
    return groups


def __decode(cmap: ChunkMap, stored: memoryview) -> np.ndarray:
    import blosc2

    if cmap.fletcher32:
        __verify_fletcher32(stored, cmap.path)
        stored = stored[:-4]
    if cmap.row_shape:
        decoded = blosc2.ndarray_from_cframe(stored)[...]
        return (
            np.ascontiguousarray(decoded)
            .view(cmap.dtype)
            .reshape((-1,) + cmap.row_shape)
        )
    schunk = blosc2.schunk_from_cframe(stored)
    if schunk.nchunks != 1:
        raise RuntimeError(
            f"Expected one Blosc2 chunk per HDF5 chunk, found {schunk.nchunks}"
        )
    return np.frombuffer(schunk.decompress_chunk(0), cmap.dtype)


def __place(
    read_plan: ReadPlan, w: int, cmap: ChunkMap, chunk: np.ndarray, out: np.ndarray
):
    windows = read_plan.windows
    row_lo, row_hi = int(windows.row_lo[w]), int(windows.row_hi[w])
    first = row_lo - (row_lo // cmap.chunk_rows) * cmap.chunk_rows
    rows = chunk[first : first + row_hi - row_lo]
    out_lo, out_hi = int(windows.out_lo[w]), int(windows.out_hi[w])
    if out_hi - out_lo == row_hi - row_lo:
        out[out_lo:out_hi] = rows
    else:
        scatter(read_plan, w, rows, out)


def __verify_fletcher32(stored: memoryview, path: str):
    """
    HDF5's fletcher32 over big-endian 16-bit words ``w`` (an odd final byte is
    the high byte of a last word), stored little-endian after the data:
    ``sum1 = Σ w_i`` and ``sum2 = Σ (n - i) w_i``, both mod 65535. HDF5 also
    accepts the checksum with the bytes of each half swapped. The words are
    summed in rows of ``FLETCHER_BLOCK``, so ``Σ i w_i`` comes from row and
    column sums and no per-word weights are built.
    """
    data = stored[:-4]
    words = np.frombuffer(data, ">u2", count=len(data) // 2).astype(np.uint16)
    n_blocks = len(words) // FLETCHER_BLOCK
    blocks = words[: n_blocks * FLETCHER_BLOCK].reshape(n_blocks, FLETCHER_BLOCK)
    tail = words[n_blocks * FLETCHER_BLOCK :].astype(np.uint64)
    acc = np.uint32 if n_blocks <= 1 << 16 else np.uint64
    rows = blocks.sum(axis=1, dtype=np.uint32).astype(np.uint64)
    cols = blocks.sum(axis=0, dtype=acc).astype(np.uint64)

    n = len(words)
    total = int(rows.sum()) + int(tail.sum())
    weighted = (
        FLETCHER_BLOCK * int(np.dot(rows, np.arange(n_blocks, dtype=np.uint64)))
        + int(np.dot(cols, np.arange(FLETCHER_BLOCK, dtype=np.uint64)))
        + int(np.dot(tail, np.arange(n - len(tail), n, dtype=np.uint64)))
    )
    if len(data) % 2:
        last = data[-1] << 8
        total += last
        weighted += n * last
        n += 1
    sum1, sum2 = total % 65535, (n * total - weighted) % 65535

    stored_sum = int.from_bytes(stored[-4:], "little")
    swapped = ((stored_sum & 0x00FF00FF) << 8) | ((stored_sum >> 8) & 0x00FF00FF)
    for candidate in (stored_sum, swapped):
        if (candidate & 0xFFFF) % 65535 == sum1 and (candidate >> 16) % 65535 == sum2:
            return
    raise OSError(f"Fletcher32 checksum mismatch in a chunk of {path}")
