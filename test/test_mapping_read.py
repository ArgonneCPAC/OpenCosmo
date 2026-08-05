from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import h5py
import numpy as np
from opencosmo.mapping.read import read_match_set

if TYPE_CHECKING:
    from pathlib import Path

# Fixed UUIDs used across tests.
_REF = UUID("00000000-0000-0000-0000-000000000001")
_A = UUID("00000000-0000-0000-0000-000000000002")
_B = UUID("00000000-0000-0000-0000-000000000003")
_C = UUID("00000000-0000-0000-0000-000000000004")


def _write_simple_slot(group: h5py.Group, name: str, data: list[int]) -> None:
    """Write a simple (index-only) slot under *group*."""
    slot = group.require_group(name)
    slot.create_dataset("index", data=np.array(data, dtype=np.int64))


def _write_aux_pair(
    aux_group: h5py.Group,
    uuid_a: UUID,
    uuid_b: UUID,
    source_data: list[int],
    target_data: list[int],
) -> None:
    """Write an auxiliary pair slot under *aux_group*."""
    pair_name = f"{uuid_a}__{uuid_b}"
    pair = aux_group.require_group(pair_name)
    _write_simple_slot(pair, "source", source_data)
    _write_simple_slot(pair, "target", target_data)


def _make_map_group(
    f: h5py.File,
    reference: UUID,
    primary_targets: dict[UUID, list[int]] | None = None,
    aux_pairs: dict[tuple[UUID, UUID], tuple[list[int], list[int]]] | None = None,
) -> h5py.Group:
    """
    Build a /map group in *f* with the given reference and optional primary/aux slots.
    Returns the /map group.
    """
    map_group = f.require_group("map")
    map_group.attrs["reference"] = str(reference)

    if primary_targets:
        primary = map_group.require_group("primary")
        for target_uuid, data in primary_targets.items():
            _write_simple_slot(primary, str(target_uuid), data)

    if aux_pairs:
        auxiliary = map_group.require_group("auxiliary")
        for (uuid_a, uuid_b), (src, tgt) in aux_pairs.items():
            _write_aux_pair(auxiliary, uuid_a, uuid_b, src, tgt)

    return map_group


class TestReferenceAbsentTwoPrimaryTargets:
    """Bug fix: reference absent + two primary targets -> match set returned."""

    def test_returns_dataset_match_set(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _B: [2, 0, 1],
                },
            )
            # Reference (_REF) is NOT in available.
            available = {_A, _B}
            result = read_match_set(map_group, available)

            assert result is not None
            assert result.reference_source == _REF
            assert set(result.primary_maps.keys()) == {_A, _B}
            assert result.aux_maps == {}


class TestReferenceAbsentOnePrimaryNoAux:
    """Reference absent, exactly one primary target, no auxiliary -> None."""

    def test_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1, 2]},
            )
            available = {_A}
            result = read_match_set(map_group, available)

            assert result is None


class TestReferenceAbsentOnePrimaryWithAux:
    """Reference absent, one primary target, one auxiliary pair -> aux retained, primary discarded."""

    def test_aux_retained_primary_discarded(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1, 2]},
                aux_pairs={(_A, _B): ([0, 1], [2, 0])},
            )
            # Reference absent; both _A and _B are available.
            available = {_A, _B}
            result = read_match_set(map_group, available)

            assert result is not None
            # Lone primary must be discarded.
            assert result.primary_maps == {}
            # Auxiliary pair must be retained.
            assert (_A, _B) in result.aux_maps


class TestReferencePresent:
    """Reference present -> primary maps populated (pre-existing behaviour)."""

    def test_primary_maps_populated(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _B: [2, 0, 1],
                },
            )
            # Reference is present this time.
            available = {_REF, _A, _B}
            result = read_match_set(map_group, available)

            assert result is not None
            assert result.reference_source == _REF
            assert set(result.primary_maps.keys()) == {_A, _B}


class TestTargetsNotInAvailableFiltered:
    """Targets not in available are filtered out."""

    def test_unavailable_target_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _B: [2, 0, 1],
                    _C: [1, 2, 0],
                },
            )
            # _C is not available.
            available = {_REF, _A, _B}
            result = read_match_set(map_group, available)

            assert result is not None
            assert _C not in result.primary_maps
            assert set(result.primary_maps.keys()) == {_A, _B}

    def test_aux_pair_with_unavailable_endpoint_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1, 2]},
                # _C is not available, so this pair must be dropped.
                aux_pairs={(_A, _C): ([0], [1])},
            )
            available = {_REF, _A}
            result = read_match_set(map_group, available)

            # Only one primary target with reference present -> primary kept,
            # but aux dropped because _C is absent.
            assert result is not None
            assert (_A, _C) not in result.aux_maps


class TestMissingReferenceAttr:
    """Missing reference attr -> returns None."""

    def test_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            # Deliberately omit the "reference" attr.
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1, 2])

            available = {_REF, _A}
            result = read_match_set(map_group, available)

            assert result is None
