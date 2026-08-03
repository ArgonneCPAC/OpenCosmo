from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import h5py

from opencosmo.header import read_header

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

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
    reference_paths : list[Path]
        All files of one representative (lightest) step. Empty ranks open this whole
        step so that they can build the same collection kind as the busy ranks
        (a single file is not a valid structure collection).
    is_structure_collection : bool
        True when each step is described by several linked files (a properties file
        containing a ``/data_linked`` group plus its linked particles/profiles), so
        the open path must build a ``StructureCollection`` rather than a plain
        ``Lightcone``.
    """

    paths: list[list[Path]]
    reference_paths: list[Path]
    is_structure_collection: bool


def partition_contiguous(weights: list[int], k: int) -> list[list[int]]:
    """
    Split a sequence of ``weights`` into ``k`` *contiguous* groups, minimizing the
    largest group sum (the classic linear-partition problem, solved optimally with
    dynamic programming).

    This is what makes each rank receive one contiguous run of redshift steps of
    roughly equal data volume, rather than an arbitrary scatter of steps that
    merely happen to balance the row counts.

    Parameters
    ----------
    weights : list[int]
        Per-item weights, in the order the items must stay in (redshift order).
    k : int
        Number of contiguous groups (ranks) to produce.

    Returns
    -------
    list[list[int]]
        ``k`` lists of item indices. Each inner list is a contiguous, ascending
        run of indices; concatenating them in order reproduces ``range(len(weights))``.
        When there are fewer items than groups, the surplus trailing groups are empty.
    """
    if k <= 0:
        raise ValueError("Number of groups must be positive")

    n = len(weights)
    if n == 0:
        return [[] for _ in range(k)]
    if k >= n:
        # One item per group for the first n groups; the rest are empty.
        return [[i] for i in range(n)] + [[] for _ in range(k - n)]

    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + weights[i]

    # dp[j][i] = minimal achievable max-group-sum when splitting the first i items
    # into j contiguous groups. split[j][i] records where the last group starts.
    inf = float("inf")
    dp: list[list[float]] = [[inf] * (n + 1) for _ in range(k + 1)]
    split: list[list[int]] = [[0] * (n + 1) for _ in range(k + 1)]
    for i in range(1, n + 1):
        dp[1][i] = prefix[i]
    for j in range(2, k + 1):
        # Need at least j items to form j non-empty groups.
        for i in range(j, n + 1):
            best = inf
            best_m = j - 1
            for m in range(j - 1, i):  # last group covers items [m, i)
                candidate = max(dp[j - 1][m], prefix[i] - prefix[m])
                if candidate < best:
                    best = candidate
                    best_m = m
            dp[j][i] = best
            split[j][i] = best_m

    groups: list[list[int]] = []
    i = n
    for j in range(k, 0, -1):
        m = split[j][i] if j > 1 else 0
        groups.append(list(range(m, i)))
        i = m
    groups.reverse()
    return groups


def plan_redshift_distribution(
    paths: list["Path"], comm: Optional["MPI.Comm"]
) -> Optional[DistributionPlan]:
    """
    Plan a redshift-based distribution of lightcone files across MPI ranks.

    Computed on rank 0 only, then broadcast to all ranks. Returns None (fallback
    sentinel) if the layout is a nested Diffsky-style lightcone (CASE B), in which
    case the caller should fall back to spatial distribution.

    Performs compatibility verification on rank 0:
    - Every file must be a lightcone (is_lightcone=True).
    - Files of a given data_type must share column names and dtypes across steps;
      different data_types may differ (structure-collection files intentionally do).
    - Classifies the layout as a plain lightcone or a lightcone structure
      collection (CASE A), or falls back (None) for anything else.

    Splits the redshift-ordered steps into contiguous chunks of roughly-equal row
    count (optimal linear partition), so each rank owns one continuous redshift
    range. Keeps same-step files together; for structure-collections, keeps linked
    datasets with their parent.

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

    # Rank 0 is the only rank that sees every file, so it does the planning and
    # compatibility verification. It must NOT raise directly: the other ranks are
    # blocked in the bcast below, and a bare raise on rank 0 would deadlock them.
    # Instead capture any error, broadcast it, and have every rank re-raise it in
    # lockstep so the failure is collective.
    result: DistributionPlan | ValueError | None = None
    if rank == 0:
        try:
            result = __compute_redshift_distribution_plan(paths, nranks)
        except ValueError as e:
            result = e

    # Broadcast the plan (None sentinel for fallback, or a ValueError) to all ranks.
    if comm:
        result = comm.bcast(result, root=0)

    if isinstance(result, ValueError):
        raise result

    return result


def __compute_redshift_distribution_plan(
    paths: list["Path"], nranks: int
) -> Optional[DistributionPlan]:
    """
    Rank-0-only computation of the distribution plan.
    """
    if not paths:
        raise ValueError("No paths provided for distribution planning")

    # Per-file metadata, grouped by step. Each entry carries the path, the row
    # count of its top-level /data (0 for particle/profile files that only have
    # linked groups), its data_type, and whether it is a properties file holding a
    # /data_linked group (the reliable on-disk signal for CASE A).
    file_info: dict[int | None, list[tuple["Path", int, str, bool]]] = {}
    # Per-data_type column signature, so different linked types may differ from one
    # another but must be self-consistent across steps.
    columns_by_type: dict[str, set[str]] = {}
    dtypes_by_type: dict[str, list[np.dtype]] = {}

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

            # Collect column info from /data and linked groups
            columns_and_dtypes = __get_columns_info(f)
            cols = set(columns_and_dtypes.keys())
            dtypes = list(columns_and_dtypes.values())

            # Per-data_type compatibility: files of the same type must share
            # columns and dtypes across steps, but different types may differ.
            if data_type not in columns_by_type:
                columns_by_type[data_type] = cols
                dtypes_by_type[data_type] = dtypes
            else:
                if cols != columns_by_type[data_type]:
                    raise ValueError(
                        f"File {path} has different columns than previous "
                        f"'{data_type}' files. Expected "
                        f"{sorted(columns_by_type[data_type])}, got {sorted(cols)}. "
                        "All lightcone files of a given data type must have "
                        "identical columns."
                    )
                if dtypes != dtypes_by_type[data_type]:
                    raise ValueError(
                        f"File {path} has different column dtypes than previous "
                        f"'{data_type}' files. All lightcone files of a given data "
                        "type must have identical column dtypes."
                    )

            # Row count from top-level /data (0 for files that only carry linked
            # groups, e.g. particles/profiles).
            row_count = __get_data_row_count(f)

            is_properties_link = (
                data_type in ("halo_properties", "galaxy_properties")
                and "data_linked" in f
            )

            file_info.setdefault(step, []).append(
                (path, row_count, data_type, is_properties_link)
            )

    is_structure_collection = __determine_collection_kind(file_info)
    if is_structure_collection is None:
        # CASE B (e.g. nested Diffsky step->type lightcone): fall back to spatial.
        return None

    # Order steps by redshift (the step index is monotonic in redshift). Each step
    # becomes one indivisible unit whose weight is the total rows across all of its
    # files, and whose paths (properties + any linked particles/profiles) travel
    # together to the same rank.
    ordered_steps = sorted(file_info.keys(), key=lambda x: (x is None, x))  # type: ignore
    step_paths: list[list["Path"]] = []
    step_weights: list[int] = []
    for step in ordered_steps:
        step_paths.append([path for path, _, _, _ in file_info[step]])
        step_weights.append(sum(rc for _, rc, _, _ in file_info[step]))

    # Split the ordered steps into nranks CONTIGUOUS chunks of roughly-equal row
    # count, so each rank owns one continuous redshift range.
    step_groups = partition_contiguous(step_weights, nranks)
    rank_files: list[list["Path"]] = []
    for group in step_groups:
        files_for_rank: list["Path"] = []
        for step_idx in group:
            files_for_rank.extend(step_paths[step_idx])
        rank_files.append(files_for_rank)

    # Reference = all files of the lightest step. Empty ranks open this whole step
    # so they can build the same collection kind as the busy ranks.
    lightest = min(range(len(step_weights)), key=lambda i: step_weights[i])
    reference_paths = list(step_paths[lightest])

    return DistributionPlan(
        paths=[list(rf) for rf in rank_files],
        reference_paths=reference_paths,
        is_structure_collection=is_structure_collection,
    )


def __determine_collection_kind(
    file_info: dict[int | None, list[tuple["Path", int, str, bool]]],
) -> Optional[bool]:
    """
    Classify the on-disk layout from the per-step file metadata.

    Returns
    -------
    False
        Plain lightcone: every step has exactly one file of a single, consistent
        data_type.
    True
        Structure collection (CASE A): each step has a properties file containing a
        /data_linked group plus one or more linked types; every step shares the same
        set of data_types.
    None
        Fallback (CASE B and anything else): the caller should use spatial
        distribution.

    Raises
    ------
    ValueError
        If the layout looks like a structure collection but is inconsistent across
        steps (missing properties file, or differing data_type sets).
    """
    # Plain lightcone: one file per step, all the same data_type.
    single_file = all(len(files) == 1 for files in file_info.values())
    single_type = (
        len({dt for files in file_info.values() for _, _, dt, _ in files}) == 1
    )
    if single_file and single_type:
        return False

    # Structure collection: multiple files per step, one of which is a
    # properties-with-data_linked file. Require every step to look the same.
    any_properties_link = any(
        is_link for files in file_info.values() for _, _, _, is_link in files
    )
    if not any_properties_link:
        # Multiple files/types per step but no properties link -> CASE B / unknown.
        return None

    type_sets = {frozenset(dt for _, _, dt, _ in files) for files in file_info.values()}
    if len(type_sets) != 1:
        raise ValueError(
            "Lightcone structure collection steps have inconsistent data_type sets: "
            f"{sorted(sorted(ts) for ts in type_sets)}. Every step must contain the "
            "same set of linked file types."
        )
    for step, files in file_info.items():
        if not any(is_link for _, _, _, is_link in files):
            raise ValueError(
                f"Lightcone structure collection step {step} has no properties file "
                "with a /data_linked group."
            )
    return True


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


def __get_data_row_count(file: h5py.File) -> int:
    """
    Row count of the top-level /data group, or 0 if the file has none (particle and
    profile files carry only linked groups). This is a per-file weight proxy; the
    planner sums it across a step's files for the total-volume estimate.
    """
    if "/data" in file:
        first_col = next(iter(file["/data"].values()), None)
        if first_col is not None:
            return first_col.shape[0]
    return 0
