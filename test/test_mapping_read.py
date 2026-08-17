from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import h5py
import numpy as np
import pytest
from opencosmo.io.discover import _read_map_layout, discover_file
from opencosmo.mapping.mapping import DatasetMatchSet, get_mapping
from opencosmo.mapping.read import read_match_set

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.io.discover import MapLayout

# Fixed UUIDs used across tests.
_REF = UUID("00000000-0000-0000-0000-000000000001")
_A = UUID("00000000-0000-0000-0000-000000000002")
_B = UUID("00000000-0000-0000-0000-000000000003")
_C = UUID("00000000-0000-0000-0000-000000000004")


def test_missing_primary_route_has_clear_error() -> None:
    match_set = DatasetMatchSet(
        reference_source=_REF,
        primary_maps={},
        aux_maps={},
        aliases={"source": _A, "target": _B},
    )

    with pytest.raises(
        ValueError,
        match="Unable to map from 'source' to 'target'.*no primary mapping route",
    ):
        get_mapping(match_set, "source", "target", np.array([0], dtype=np.int64))


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
    pair.create_dataset("source", data=np.array(source_data, dtype=np.int64))
    pair.create_dataset("target", data=np.array(target_data, dtype=np.int64))


def _make_map_group(
    f: h5py.File,
    reference: UUID,
    primary_targets: dict[UUID, list[int]] | None = None,
    aux_pairs: dict[tuple[UUID, UUID], tuple[list[int], list[int]]] | None = None,
) -> tuple[h5py.Group, MapLayout]:
    """
    Build a /map group in *f* with the given reference and optional primary/aux slots.
    Returns the /map group and its parsed MapLayout (via the real discovery parser),
    so that tests exercise the actual name<->UUID pairing rather than hand-constructing
    a layout that might diverge from what the reader sees.
    """
    map_group = f.require_group("map")
    map_group.attrs["reference"] = str(reference)
    map_group.attrs["format_version"] = 1

    if primary_targets:
        primary = map_group.require_group("primary")
        for target_uuid, data in primary_targets.items():
            _write_simple_slot(primary, str(target_uuid), data)

    if aux_pairs:
        auxiliary = map_group.require_group("auxiliary")
        for (uuid_a, uuid_b), (src, tgt) in aux_pairs.items():
            _write_aux_pair(auxiliary, uuid_a, uuid_b, src, tgt)

    layout = _read_map_layout("/map", map_group)
    return map_group, layout


class TestReferenceAbsentTwoPrimaryTargets:
    """Bug fix: reference absent + two primary targets -> match set returned."""

    def test_returns_dataset_match_set(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _B: [2, 0, 1],
                },
            )
            # Reference (_REF) is NOT in available.
            available = {_A, _B}
            result = read_match_set(map_group, layout, available)

            assert result is not None
            assert result.reference_source == _REF
            assert set(result.primary_maps.keys()) == {_A, _B}
            assert result.aux_maps == {}


class TestReferenceAbsentOnePrimaryNoAux:
    """Reference absent, exactly one primary target, no auxiliary -> None."""

    def test_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1, 2]},
            )
            available = {_A}
            result = read_match_set(map_group, layout, available)

            assert result is None


class TestAuxiliaryRequiresPrimaries:
    """Every auxiliary endpoint must have a corresponding primary map."""

    def test_missing_primary_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            with pytest.raises(ValueError, match="without corresponding primaries"):
                _make_map_group(
                    f,
                    reference=_REF,
                    primary_targets={_A: [0, 1, 2]},
                    aux_pairs={(_A, _B): ([0, 1], [2, 0])},
                )


class TestMapArrayValidation:
    @pytest.mark.parametrize(
        ("data", "message"),
        (
            (np.array([[0, 1]], dtype=np.int64), "one-dimensional"),
            (np.array([0.0, 1.0]), "integer dtype"),
        ),
    )
    def test_primary_array_shape_and_dtype(self, tmp_path: Path, data, message) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            slot = map_group.require_group(f"primary/{_A}")
            slot.create_dataset("index", data=data)

            with pytest.raises(ValueError, match=message):
                _read_map_layout("/map", map_group)

    def test_chunked_mapping_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            slot = map_group.require_group(f"primary/{_A}")
            slot.create_dataset("start", data=np.array([0, 1], dtype=np.int64))
            slot.create_dataset("size", data=np.array([1, 1], dtype=np.int64))

            with pytest.raises(ValueError, match="chunked.*not supported"):
                _read_map_layout("/map", map_group)

        layout = discover_file(path)
        assert layout.error is not None
        assert "chunked" in layout.error

    def test_slot_cannot_mix_simple_and_chunked_arrays(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            slot = map_group.require_group(f"primary/{_A}")
            slot.create_dataset("index", data=np.array([0, 1], dtype=np.int64))
            slot.create_dataset("start", data=np.array([0, 1], dtype=np.int64))
            slot.create_dataset("size", data=np.array([1, 1], dtype=np.int64))

            with pytest.raises(ValueError, match="chunked.*not supported"):
                _read_map_layout("/map", map_group)

    @pytest.mark.parametrize("version", (2, "1"))
    def test_unsupported_format_version(self, tmp_path: Path, version) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = version

            with pytest.raises(ValueError, match="only version 1 is supported"):
                _read_map_layout("/map", map_group)

    def test_auxiliary_sides_require_equal_lengths(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            with pytest.raises(ValueError, match="source and target.*same length"):
                _make_map_group(
                    f,
                    reference=_REF,
                    primary_targets={_A: [0, 1], _B: [1, 0]},
                    aux_pairs={(_A, _B): ([0, 1], [1])},
                )

    @pytest.mark.parametrize(
        ("side", "data", "message"),
        (
            ("source", np.array([[0, 1]], dtype=np.int64), "one-dimensional"),
            ("target", np.array([0.0, 1.0]), "integer dtype"),
        ),
    )
    def test_auxiliary_array_shape_and_dtype(
        self, tmp_path: Path, side, data, message
    ) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, _ = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1], _B: [1, 0]},
                aux_pairs={(_A, _B): ([0, 1], [1, 0])},
            )
            pair = map_group[f"auxiliary/{_A}__{_B}"]
            del pair[side]
            pair.create_dataset(side, data=data)

            with pytest.raises(ValueError, match=message):
                _read_map_layout("/map", map_group)

    @pytest.mark.parametrize("side", ("source", "target"))
    def test_auxiliary_side_must_be_direct_dataset(self, tmp_path: Path, side) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, _ = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1], _B: [1, 0]},
                aux_pairs={(_A, _B): ([0, 1], [1, 0])},
            )
            pair = map_group[f"auxiliary/{_A}__{_B}"]
            del pair[side]
            _write_simple_slot(pair, side, [0, 1])

            with pytest.raises(ValueError, match="expected a dataset"):
                _read_map_layout("/map", map_group)

    @pytest.mark.parametrize(
        ("source", "target", "message"),
        (
            ([-1, 1], [0, 1], "source indices must be non-negative"),
            ([0, 1], [-1, 1], "target indices must be non-negative"),
            ([0, 0], [0, 1], "source indices must be unique"),
            ([0, 1], [0, 0], "target indices must be unique"),
        ),
    )
    def test_auxiliary_values_are_one_to_one(
        self, tmp_path: Path, source, target, message
    ) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1], _B: [1, 0]},
                aux_pairs={(_A, _B): (source, target)},
            )

            with pytest.raises(ValueError, match=message):
                read_match_set(map_group, layout, {_REF, _A, _B})


class TestMapUUIDCollisions:
    def test_primary_spellings_cannot_resolve_to_same_uuid(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1])
            _write_simple_slot(primary, _A.hex, [0, 1])

            with pytest.raises(ValueError, match="duplicate primary mapping"):
                _read_map_layout("/map", map_group)

    def test_auxiliary_spellings_cannot_resolve_to_same_pair(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1])
            _write_simple_slot(primary, str(_B), [1, 0])
            auxiliary = map_group.require_group("auxiliary")
            _write_aux_pair(auxiliary, _A, _B, [0], [1])
            reversed_pair = auxiliary.require_group(f"{_B.hex}__{_A.hex}")
            reversed_pair.create_dataset("source", data=np.array([1], dtype=np.int64))
            reversed_pair.create_dataset("target", data=np.array([0], dtype=np.int64))

            with pytest.raises(ValueError, match="duplicate auxiliary mapping pair"):
                _read_map_layout("/map", map_group)


class TestReferencePresent:
    """Reference present -> primary maps populated (pre-existing behaviour)."""

    def test_primary_maps_populated(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _B: [2, 0, 1],
                },
            )
            # Reference is present this time.
            available = {_REF, _A, _B}
            result = read_match_set(map_group, layout, available)

            assert result is not None
            assert result.reference_source == _REF
            assert set(result.primary_maps.keys()) == {_A, _B}

    def test_auxiliary_maps_resolve_direct_datasets(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1], _B: [1, 0]},
                aux_pairs={(_A, _B): ([2, 3], [4, 5])},
            )

            result = read_match_set(map_group, layout, {_REF, _A, _B})

            assert result is not None
            source, target = result.aux_maps[(_A, _B)]
            assert source.name.endswith("/source")
            assert target.name.endswith("/target")
            np.testing.assert_array_equal(source[:], [2, 3])
            np.testing.assert_array_equal(target[:], [4, 5])


class TestTargetsNotInAvailableFiltered:
    """Targets not in available are filtered out."""

    def test_unavailable_target_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
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
            result = read_match_set(map_group, layout, available)

            assert result is not None
            assert _C not in result.primary_maps
            assert set(result.primary_maps.keys()) == {_A, _B}

    def test_aux_pair_with_unavailable_endpoint_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _C: [1, 2, 0],
                },
                # _C is not available, so this pair must be dropped.
                aux_pairs={(_A, _C): ([0], [1])},
            )
            available = {_REF, _A}
            result = read_match_set(map_group, layout, available)

            # Only one primary target with reference present -> primary kept,
            # but aux dropped because _C is absent.
            assert result is not None
            assert (_A, _C) not in result.aux_maps


class TestMissingReferenceAttr:
    """Missing reference attr is a discovery-time error, not a read_match_set None return."""

    def test_read_map_layout_raises_on_missing_reference(self, tmp_path: Path) -> None:
        """_read_map_layout raises ValueError mentioning 'reference' when the attr is absent."""
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["format_version"] = 1
            # Deliberately omit the "reference" attr.
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1, 2])

            with pytest.raises(ValueError, match="reference"):
                _read_map_layout("/map", map_group)

    def test_discover_file_yields_error_on_missing_reference(
        self, tmp_path: Path
    ) -> None:
        """discover_file returns a FileLayout with a non-None error for a missing reference."""
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["format_version"] = 1
            # Deliberately omit the "reference" attr.
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1, 2])

        layout = discover_file(path)
        assert layout.error is not None
        assert "reference" in layout.error


class TestNonCanonicalSlotName:
    """Regression guard: a /primary slot whose on-disk name is a non-canonical UUID
    spelling (e.g. uppercase or 32-char hex without hyphens) must still resolve
    correctly through the layout.  This guards against anyone later 'simplifying'
    the reader to rebuild names with f'{uuid}' instead of using the recorded name."""

    def test_uppercase_uuid_slot_resolves(self, tmp_path: Path) -> None:
        # Verify that uuid.UUID accepts the uppercase spelling we use on disk.
        uppercase_name = str(_A).upper()
        assert UUID(uppercase_name) == _A, "precondition: UUID() accepts uppercase"

        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1

            # Write the primary slot with an uppercase (non-canonical) name.
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, uppercase_name, [0, 1, 2])

            layout = _read_map_layout("/map", map_group)

            # The layout must record the verbatim on-disk name, not str(_A).
            assert len(layout.primary_slots) == 1
            slot_name, target_uuid = layout.primary_slots[0]
            assert slot_name == uppercase_name
            assert target_uuid == _A

            # read_match_set must resolve the live handle correctly via the recorded name.
            available = {_REF, _A}
            result = read_match_set(map_group, layout, available)

            assert result is not None
            assert _A in result.primary_maps
