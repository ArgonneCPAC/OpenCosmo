from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import h5py

from opencosmo.header import read_header

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.mpi import MPI


@dataclass(frozen=True)
class GroupLayout:
    """Frozen layout of a single /data group within a file."""

    path: str
    """In-file path of the group holding /data, e.g. "/" or "/scidac1" or "/halo_properties"."""

    header_path: str
    """In-file path of the group's governing header (nearest enclosing /header)."""

    header: OpenCosmoHeader
    """The header object read from the governing header group."""

    column_names: tuple[str, ...]
    """Sorted tuple of column names under this group's /data."""

    column_dtypes: tuple[str, ...]
    """str(dtype) for each column, same order as column_names."""

    row_count: int
    """Number of rows in the first column, or 0 if no /data group."""

    has_index: bool
    """Whether an /index group exists as a sibling of /data."""

    linked_target_names: tuple[str, ...]
    """Sorted tuple of linked target name prefixes (from /data_linked _start/_size suffixes)."""


@dataclass(frozen=True)
class FileLayout:
    """Frozen layout of a complete file."""

    path: Path
    """Path to the file."""

    groups: tuple[GroupLayout, ...]
    """Sorted tuple of GroupLayout objects found in this file."""

    error: Optional[str] = None
    """Error message if discovery failed, None if successful."""


def _make_group_map(
    group: h5py.File | h5py.Group, prefix: str = ""
) -> dict[str, h5py.File | h5py.Group | h5py.Dataset]:
    """
    Build a flat map of all h5py objects in a file/group tree.
    Keys are in-file paths (e.g., "/header", "/scidac1/data").
    """
    index = {}
    for key, item in group.items():
        path = f"{prefix}/{key}"
        index[path] = item
        if isinstance(item, h5py.Group):
            index.update(_make_group_map(item, path))
    return index


def _header_scope(header_path: str) -> str:
    """Group path a "/header" governs (its parent): "/header" -> "/"."""
    return header_path.rsplit("/header", 1)[0] or "/"


def _iter_ancestors(group_path: str) -> Iterator[str]:
    """Yield ``group_path`` then each ancestor up to the root, deepest first."""
    yield group_path
    while group_path != "/":
        group_path = group_path.rsplit("/", 1)[0] or "/"
        yield group_path


def discover_file(path: Path) -> FileLayout:
    """
    Walk a file once and produce a frozen, picklable layout with no live h5py handles.

    Reuses the walk logic from iopen's __make_group_map. Steps:
    1. Open with h5py.File(path, "r") in a with block.
    2. Build the group map.
    3. Find every /header group and every /data group.
    4. Read each header once, locally, via read_header.__wrapped__ (no world-comm
       broadcast), keyed by the group scope it governs.
    5. For each /data group, resolve its governing header as the nearest enclosing
       scope on the group's own ancestry.
    6. Extract column metadata (names, dtypes, row_count) for each group.
    7. Check for /index and /data_linked groups.
    8. Sort groups by path for determinism.

    On any structural failure (file not an OpenCosmo file, malformed header, missing /data),
    return FileLayout(path=path, groups=(), error=<message>) — never raise.
    """
    try:
        with h5py.File(path, "r") as f:
            file_map = _make_group_map(f)

            # Find all header and data groups.
            header_groups = sorted([k for k in file_map.keys() if k.endswith("header")])
            data_groups = sorted([k for k in file_map.keys() if k.endswith("/data")])

            if not header_groups:
                return FileLayout(
                    path=path, groups=(), error="No header groups found in file"
                )

            if not data_groups:
                return FileLayout(
                    path=path, groups=(), error="No data groups found in file"
                )

            # Read every header exactly once, keyed by the group scope it governs.
            # A header at "/scidac1/header" governs the "/scidac1" scope; the root
            # "/header" governs "/". A data group's governing header is then the
            # nearest of these scopes on its own ancestry — no rescan of all headers,
            # and a header shared by several data groups is parsed a single time.
            read_header_local = read_header.__wrapped__  # type: ignore[attr-defined]
            scope_to_header: dict[str, tuple[str, OpenCosmoHeader]] = {}
            for header_path in header_groups:
                try:
                    header = read_header_local(file_map[header_path].parent)
                except (KeyError, ValueError, TypeError) as e:
                    return FileLayout(
                        path=path,
                        groups=(),
                        error=f"Malformed header at {header_path}: {e}",
                    )
                scope_to_header[_header_scope(header_path)] = (header_path, header)

            group_layouts = []
            for data_path in data_groups:
                # The owning group is the parent of /data ("/scidac1/data" -> "/scidac1").
                group_path = data_path.rsplit("/data", 1)[0] or "/"

                # Governing header = nearest enclosing scope on this group's ancestry.
                # The root scope "/" is an ancestor of every group, so a valid file
                # with a top-level header always resolves.
                governing_scope = next(
                    (a for a in _iter_ancestors(group_path) if a in scope_to_header),
                    None,
                )
                if governing_scope is None:
                    return FileLayout(
                        path=path,
                        groups=(),
                        error=f"No header governs data group {data_path}",
                    )
                governing_header_path, header = scope_to_header[governing_scope]

                # Extract column metadata for this data group.
                data_prefix = data_path if data_path.endswith("/") else data_path + "/"
                columns_in_data = sorted(
                    [
                        k.split(data_prefix, 1)[1]
                        for k in file_map.keys()
                        if k.startswith(data_prefix)
                        and isinstance(file_map[k], h5py.Dataset)
                        and not k.endswith("/index")
                        and "header" not in k
                    ]
                )

                column_names = tuple(columns_in_data)
                column_dtypes = tuple(
                    str(file_map[f"{data_path}/{col}"].dtype) for col in columns_in_data
                )

                # Row count from the first column, or 0 if no columns.
                row_count = 0
                if columns_in_data:
                    first_col = file_map[f"{data_path}/{columns_in_data[0]}"]
                    row_count = first_col.shape[0]

                # Check for index group.
                group_parent = (
                    group_path if group_path.endswith("/") else group_path + "/"
                )
                index_path = f"{group_parent}index"
                has_index = index_path in file_map

                # Check for data_linked group and extract linked target names.
                linked_target_names_set = set()
                data_linked_path = f"{group_parent}data_linked"
                if data_linked_path in file_map:
                    data_linked_group = file_map[data_linked_path]
                    if isinstance(data_linked_group, h5py.Group):
                        for key in data_linked_group.keys():
                            if key.endswith("_start"):
                                target_name = key.rsplit("_start", 1)[0]
                                linked_target_names_set.add(target_name)
                            elif key.endswith("_size"):
                                target_name = key.rsplit("_size", 1)[0]
                                linked_target_names_set.add(target_name)

                linked_target_names = tuple(sorted(linked_target_names_set))

                group_layouts.append(
                    GroupLayout(
                        path=group_path,
                        header_path=governing_header_path,
                        header=header,
                        column_names=column_names,
                        column_dtypes=column_dtypes,
                        row_count=row_count,
                        has_index=has_index,
                        linked_target_names=linked_target_names,
                    )
                )

            # Sort groups by path for determinism.
            sorted_groups = tuple(sorted(group_layouts, key=lambda g: g.path))
            return FileLayout(path=path, groups=sorted_groups)

    except Exception as e:
        return FileLayout(path=path, groups=(), error=str(e))


def discover_all(
    paths: list[Path],
    comm: Optional["MPI.Comm"] = None,
) -> tuple[FileLayout, ...]:
    """
    Discover metadata from all files, distributing the walk across MPI ranks.

    Rank ``i`` walks ``paths[i::nranks]`` (round-robin), then a single
    ``comm.allgather`` assembles the full list on every rank. The result is sorted
    by path, so every rank holds a byte-identical tuple. This one path covers every
    case uniformly: serial (no comm), a single file (only rank 0's slice is
    non-empty), and fewer files than ranks (surplus ranks contribute nothing).

    The only irreducible special case is the absence of a communicator: with no
    ``comm`` there is nothing to allgather, so the local walk is the final result.
    """
    rank = comm.Get_rank() if comm is not None else 0
    nranks = comm.Get_size() if comm is not None else 1

    my_layouts = [discover_file(p) for p in paths[rank::nranks]]

    if comm is not None:
        my_layouts = [
            layout
            for rank_layouts in comm.allgather(my_layouts)
            for layout in rank_layouts
        ]

    return tuple(sorted(my_layouts, key=lambda fl: str(fl.path)))


def group_data_type(group: GroupLayout) -> str:
    """Return the data_type from the group's header."""
    return str(group.header.file.data_type)


def is_lightcone_group(group: GroupLayout) -> bool:
    """Return True if the group's header marks it as a lightcone."""
    return bool(group.header.file.is_lightcone)


def has_linked_targets(group: GroupLayout) -> bool:
    """Return True if the group has any linked target names."""
    return len(group.linked_target_names) > 0


def is_healpix_map_group(group: GroupLayout) -> bool:
    """Return True if the group's data_type is healpix_map."""
    return group_data_type(group) == "healpix_map"


def is_particle_group(group: GroupLayout) -> bool:
    """Return True if the group's data_type contains 'particle'."""
    return "particle" in group_data_type(group)


def is_properties_group(group: GroupLayout) -> bool:
    """Return True if the group's data_type is halo_properties or galaxy_properties."""
    return group_data_type(group) in ("halo_properties", "galaxy_properties")


def header_scopes(
    layouts: tuple[FileLayout, ...],
) -> dict[tuple[str, str], list[GroupLayout]]:
    """
    Bucket every GroupLayout across all non-errored files by (str(file.path), group.header_path).

    Returns a dict mapping (file_path_str, header_path) -> list[GroupLayout].
    This is the grouping key for reconstructing composition (nesting logic).
    """
    scopes: dict[tuple[str, str], list[GroupLayout]] = {}
    for file_layout in layouts:
        if file_layout.error is not None:
            continue
        for group in file_layout.groups:
            key = (str(file_layout.path), group.header_path)
            if key not in scopes:
                scopes[key] = []
            scopes[key].append(group)
    return scopes
