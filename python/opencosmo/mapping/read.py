from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
from opencosmo.mapping.mapping import DatasetMatchSet

if TYPE_CHECKING:
    from uuid import UUID

    from opencosmo.mapping.mapping import H5pyIndex


def read_index_group(group: h5py.Group) -> "H5pyIndex":
    """
    Resolve a map slot into an H5pyIndex.

    A slot is either simple (an ``index`` dataset) or chunked (a ``start``+``size``
    pair). This function returns the live h5py.Dataset objects (never slicing them),
    allowing the caller to defer reads and slicing to the point of use.

    Parameters
    ----------
    group : h5py.Group
        The slot group, expected to contain either "index" or ("start" and "size").

    Returns
    -------
    H5pyIndex
        Either the "index" dataset, or a tuple of ("start", "size") datasets.
    """
    if "index" in group:
        return group["index"]  # type: ignore[return-value]
    return (group["start"], group["size"])  # type: ignore[return-value]


def read_match_set(group: h5py.Group, available: "set[UUID]") -> DatasetMatchSet | None:
    """
    Resolve a live /map group into a DatasetMatchSet.

    Walks the /primary and /auxiliary structure, building primary and auxiliary
    maps from live h5py handles. Skips any primary target or auxiliary pair whose
    endpoints are not all in ``available``.

    Primary maps are kept for every available target regardless of whether the
    reference dataset itself was opened. When the reference is absent, at least
    two primary targets must remain: target-to-target routing composes two primary
    index arrays through the reference's on-disk index without ever touching the
    reference dataset, but a lone target has nothing to compose with and is
    discarded. Auxiliary pairs whose own two endpoints are both present are always
    retained.

    ``reference_source`` always carries the reference recorded in the file, whether
    or not that dataset was opened. It names which dataset the primary maps are
    keyed against; substituting some other endpoint when the reference is absent
    would claim a routing that does not exist on disk.

    Returns None if nothing at all survives the availability check.

    Parameters
    ----------
    group : h5py.Group
        The /map group with attrs "reference" (UUID) and "format_version".
    available : set[UUID]
        UUIDs of datasets that are present in the open set.

    Returns
    -------
    DatasetMatchSet | None
        The resolved match set with available endpoints only, or None if no maps
        remain after filtering.
    """
    from uuid import UUID

    # Read the reference UUID from attrs.
    reference_str = group.attrs.get("reference")
    if reference_str is None:
        return None
    if isinstance(reference_str, bytes):
        reference_str = reference_str.decode("utf-8")
    reference = UUID(reference_str)

    # Build primary maps for every available target unconditionally.
    # If the reference itself is absent, a lone target cannot route anywhere
    # (target-to-target routing requires at least two primary maps to compose),
    # so discard the whole primary set in that case.
    primary_maps: dict[UUID, H5pyIndex] = {}
    primary_group = group.get("primary")
    if isinstance(primary_group, h5py.Group):
        for name, slot in primary_group.items():
            target_uuid = UUID(name)
            if target_uuid in available and isinstance(slot, h5py.Group):
                primary_maps[target_uuid] = read_index_group(slot)

    if reference not in available and len(primary_maps) < 2:
        primary_maps = {}

    # Process auxiliary pairs, keeping only those whose both endpoints are available.
    aux_maps: dict[tuple[UUID, UUID], tuple[H5pyIndex, H5pyIndex]] = {}
    aux_group = group.get("auxiliary")
    if isinstance(aux_group, h5py.Group):
        for name, pair in aux_group.items():
            # Parse the "<uuid_a>__<uuid_b>" name.
            parts = name.split("__")
            if len(parts) != 2:
                continue
            uuid_a = UUID(parts[0])
            uuid_b = UUID(parts[1])

            # Skip if either endpoint is not available.
            if uuid_a not in available or uuid_b not in available:
                continue

            if not isinstance(pair, h5py.Group):
                continue

            # Read the source and target index groups.
            source_group = pair.get("source")
            target_group = pair.get("target")
            if not isinstance(source_group, h5py.Group) or not isinstance(
                target_group, h5py.Group
            ):
                continue

            source_index = read_index_group(source_group)
            target_index = read_index_group(target_group)
            aux_maps[(uuid_a, uuid_b)] = (source_index, target_index)

    if not primary_maps and not aux_maps:
        return None

    return DatasetMatchSet(
        reference_source=reference,
        primary_maps=primary_maps,
        aux_maps=aux_maps,
    )
