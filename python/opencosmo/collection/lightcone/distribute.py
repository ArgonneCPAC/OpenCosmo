from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np

from opencosmo.header import read_header

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.mpi import MPI


@dataclass(frozen=True)
class DistributionPlan:
    """
    Plan for distributing lightcone files across MPI ranks by redshift step.

    Attributes
    ----------
    paths : list[list[Path]]
        Per-rank list of assigned file paths. Index is rank, value is list of paths
        that rank should open. Some ranks may have empty lists.
    reference_path : Path
        A single file to open on empty ranks (provides schema reference).
    """

    paths: list[list[Path]]
    reference_path: Path


def plan_redshift_distribution(
    paths: list["Path"], comm: Optional["MPI.Comm"]
) -> Optional[DistributionPlan]:
    """
    Plan a redshift-based distribution of lightcone files across MPI ranks.

    Computed on rank 0 only, then broadcast to all ranks. Returns None (fallback
    sentinel) if the files are nested lightcones (multiple types per step),
    in which case the caller should fall back to spatial distribution.

    Performs compatibility verification on rank 0:
    - Every file must be a lightcone (is_lightcone=True).
    - All files must share the same column names, dtypes, and data_type.
    - Detects nested lightcones (multiple dataset types per step).

    Uses greedy longest-processing-time bin-packing to assign files to ranks.
    Keeps same-step files together; for structure-collections, keeps linked datasets
    with their parent.

    Parameters
    ----------
    paths : list[Path]
        List of file paths to distribute.
    comm : Optional[MPI.Comm]
        MPI communicator, or None for serial operation.

    Returns
    -------
    DistributionPlan or None
        A plan assigning files to ranks, or None if fallback (nested lightcone) detected.
        Broadcast to all ranks.
    """
    rank = comm.Get_rank() if comm else 0
    nranks = comm.Get_size() if comm else 1

    plan = None
    if rank == 0:
        plan = __compute_redshift_distribution_plan(paths, nranks)

    # Broadcast the plan (or None) to all ranks
    if comm:
        plan = comm.bcast(plan, root=0)

    return plan


def __compute_redshift_distribution_plan(
    paths: list["Path"], nranks: int
) -> Optional[DistributionPlan]:
    """
    Rank-0-only computation of the distribution plan.
    """
    if not paths:
        raise ValueError("No paths provided for distribution planning")

    # Read metadata from each file and group by step
    file_info: dict[int | None, list[tuple["Path", int, str]]] = {}
    all_columns: set[str] | None = None
    all_dtypes: list[np.dtype] | None = None
    data_type_set: set[str] = set()

    # read_header is decorated with @broadcast_read, which fires a world-comm
    # bcast on EVERY call. Since this planner runs inside a rank-0-only block,
    # that stray collective would desynchronize the ranks (rank 0 issues N
    # header-bcasts, other ranks issue none, and the plan bcast pairs with the
    # wrong send). Call the undecorated inner function to read locally instead.
    read_header_local = read_header.__wrapped__  # type: ignore[attr-defined]

    for path in paths:
        with h5py.File(path, "r") as f:
            header = read_header_local(f)

            # Verify it's a lightcone
            if not header.file.is_lightcone:
                raise ValueError(
                    f"File {path} is not marked as a lightcone (is_lightcone=False). "
                    "Redshift-split mode requires all files to be lightcones."
                )

            step: int | None = header.file.step
            data_type = header.file.data_type
            data_type_set.add(data_type)

            # Collect column info from /data and linked groups
            columns_and_dtypes = __get_columns_info(f)
            cols = list(columns_and_dtypes.keys())
            dtypes = list(columns_and_dtypes.values())

            if all_columns is None:
                all_columns = set(cols)
            if all_dtypes is None:
                all_dtypes = dtypes

            # Verify columns match
            if set(cols) != all_columns:
                raise ValueError(
                    f"File {path} has different columns than previous files. "
                    f"Expected {sorted(all_columns)}, got {sorted(cols)}. "
                    "All lightcone files must have identical columns."
                )
            if dtypes != all_dtypes:
                raise ValueError(
                    f"File {path} has different column dtypes than previous files. "
                    "All lightcone files must have identical column dtypes."
                )

            # Get row count from first column
            first_col_key = next(iter(columns_and_dtypes.keys()))
            row_count = __get_row_count(f, first_col_key)

            if step not in file_info:
                file_info[step] = []
            file_info[step].append((path, row_count, data_type))

    # Detect nested lightcones (multiple types per step)
    for step, file_list in file_info.items():
        types_in_step = set(data_type for _, _, data_type in file_list)
        if len(types_in_step) > 1:
            # Nested lightcone detected: fallback to spatial distribution
            return None

    # Build flat list of (path, row_count) pairs, maintaining step grouping
    files_to_distribute: list[tuple["Path", int, int | None]] = []
    for step in sorted(file_info.keys(), key=lambda x: (x is None, x)):  # type: ignore
        for path, row_count, _ in file_info[step]:
            files_to_distribute.append((path, row_count, step))

    # Greedy bin-packing by row count (descending)
    files_to_distribute.sort(key=lambda x: x[1], reverse=True)

    # Initialize rank bins with total row count
    rank_totals: list[int] = [0] * nranks
    rank_files: list[list["Path"]] = [[] for _ in range(nranks)]

    for path, row_count, step in files_to_distribute:
        # Assign to rank with least total rows
        min_rank = np.argmin(rank_totals)
        rank_files[min_rank].append(path)
        rank_totals[min_rank] += row_count

    # Choose reference path (smallest file)
    reference_path = min(paths, key=lambda p: __get_file_row_count(p))

    return DistributionPlan(
        paths=[list(rf) for rf in rank_files], reference_path=reference_path
    )


def __get_columns_info(file: h5py.File) -> dict[str, np.dtype]:
    """
    Extract column names and dtypes from /data and linked groups.
    Returns dict mapping column key to dtype.
    """
    columns = {}

    # Main /data group
    if "/data" in file:
        for name, item in file["/data"].items():
            if isinstance(item, h5py.Dataset):
                columns[f"/data/{name}"] = item.dtype

    # Linked groups (e.g., /halo_properties/data, /galaxy_properties/data)
    for group_name in file.keys():
        if group_name in ("data", "header", "index"):
            continue
        if isinstance(file[group_name], h5py.Group):
            data_group = file[group_name].get("data")
            if data_group is not None and isinstance(data_group, h5py.Group):
                for name, item in data_group.items():
                    if isinstance(item, h5py.Dataset):
                        columns[f"/{group_name}/data/{name}"] = item.dtype

    return columns


def __get_row_count(file: h5py.File, column_key: str) -> int:
    """
    Get the row count from a single column dataset.
    column_key is like "/data/mass" or "/halo_properties/data/mass".
    """
    # Strip leading / and navigate
    parts = column_key.lstrip("/").split("/")
    obj = file
    for part in parts:
        obj = obj[part]
    return obj.shape[0]


def __get_file_row_count(path: "Path") -> int:
    """
    Get the total row count from a file (sum of all data groups).
    """
    total = 0
    with h5py.File(path, "r") as f:
        # Main /data group
        if "/data" in f:
            first_col = next(iter(f["/data"].values()), None)
            if first_col is not None:
                total += first_col.shape[0]
    return total
