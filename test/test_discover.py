from __future__ import annotations

import pickle
import random
from pathlib import Path  # noqa: TC003

import h5py
import pytest
from opencosmo.header import OpenCosmoHeader
from opencosmo.io.discover import (
    discover_all,
    discover_file,
    group_data_type,
    has_linked_targets,
    header_scopes,
    is_healpix_map_group,
    is_lightcone_group,
    is_particle_group,
    is_properties_group,
)


def _read_file_expected_layout(path: Path) -> dict:
    """
    Read a file with h5py and compute expected layout values.
    Returns dict with keys: column_names, column_dtypes, row_count, has_index, linked_target_names.
    """
    with h5py.File(path, "r") as f:
        # Find data group (handle both "/" and nested cases)
        data_groups = []
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                if "data" in f[key]:
                    data_groups.append("/" + key if key != "/" else "/")

        if not data_groups:
            if "data" in f:
                data_groups = ["/"]

        result = {}
        for group_key in data_groups:
            group = f[group_key] if group_key != "/" else f

            # Get data group
            if "data" in group:
                data_grp = group["data"]
                columns_in_data = sorted(
                    [
                        k
                        for k in data_grp.keys()
                        if isinstance(data_grp[k], h5py.Dataset)
                    ]
                )
                column_names = tuple(columns_in_data)
                column_dtypes = tuple(
                    str(data_grp[col].dtype) for col in columns_in_data
                )
                row_count = (
                    data_grp[columns_in_data[0]].shape[0] if columns_in_data else 0
                )

                # Check for index
                has_index = "index" in group

                # Check for linked targets
                linked_target_names_set = set()
                if "data_linked" in group:
                    data_linked = group["data_linked"]
                    for key in data_linked.keys():
                        if key.endswith("_start"):
                            target_name = key.rsplit("_start", 1)[0]
                            linked_target_names_set.add(target_name)
                        elif key.endswith("_size"):
                            target_name = key.rsplit("_size", 1)[0]
                            linked_target_names_set.add(target_name)

                linked_target_names = tuple(sorted(linked_target_names_set))

                result[group_key if group_key != "" else "/"] = {
                    "column_names": column_names,
                    "column_dtypes": column_dtypes,
                    "row_count": row_count,
                    "has_index": has_index,
                    "linked_target_names": linked_target_names,
                }

        return result


class TestDiscoverSingleFiles:
    """Test single-file discovery for various dataset types."""

    @pytest.mark.parametrize(
        "filename,is_lc",
        [
            ("haloproperties.hdf5", False),  # snapshot
            ("test_map.hdf5", False),  # healpix map
            ("lj_487.hdf5", False),  # diffsky
        ],
    )
    def test_discover_single_files(
        self, filename: str, is_lc: bool, snapshot_path, map_path, diffsky_path
    ):
        """Test discovery of single-file datasets."""
        if filename == "haloproperties.hdf5":
            path = snapshot_path / filename
        elif filename == "test_map.hdf5":
            path = map_path / filename
        else:
            path = diffsky_path / filename

        layout = discover_file(path)

        # No errors
        assert layout.error is None, f"Unexpected error: {layout.error}"

        # At least one group
        assert len(layout.groups) > 0

        # Each group matches what we read from the file
        expected_by_group = _read_file_expected_layout(path)

        for group in layout.groups:
            # Group path should be in expected data
            assert group.path in expected_by_group or "/" in expected_by_group

            group_key = group.path if group.path in expected_by_group else "/"
            expected = expected_by_group[group_key]

            # Check column names and dtypes
            assert group.column_names == expected["column_names"], (
                f"Column names mismatch for {group.path}"
            )
            assert group.column_dtypes == expected["column_dtypes"], (
                f"Column dtypes mismatch for {group.path}"
            )

            # Check row count
            assert group.row_count == expected["row_count"], (
                f"Row count mismatch for {group.path}"
            )

            # Check index
            assert group.has_index == expected["has_index"], (
                f"has_index mismatch for {group.path}"
            )

            # Check linked targets
            assert group.linked_target_names == expected["linked_target_names"], (
                f"linked_target_names mismatch for {group.path}"
            )

            # Header should be an OpenCosmoHeader
            assert isinstance(group.header, OpenCosmoHeader)

    def test_discover_lightcone_file(self, lightcone_path):
        """Test discovery of a lightcone file."""
        path = lightcone_path / "step_600" / "haloproperties.hdf5"
        layout = discover_file(path)

        assert layout.error is None
        assert len(layout.groups) > 0

        group = layout.groups[0]
        assert is_lightcone_group(group)
        assert has_linked_targets(group)

        # Verify expected structure
        expected = _read_file_expected_layout(path)["/"]
        assert group.column_names == expected["column_names"]
        assert group.row_count == expected["row_count"]
        assert group.has_index == expected["has_index"]

    def test_discover_healpix_map(self, map_path):
        """Test discovery of HEALPix map file."""
        path = map_path / "test_map.hdf5"
        layout = discover_file(path)

        assert layout.error is None
        assert len(layout.groups) > 0

        group = layout.groups[0]
        assert is_healpix_map_group(group)
        assert not group.has_index  # HEALPix maps don't have index

    def test_discover_no_h5py_leaks(self, snapshot_path):
        """Test that no h5py handles leak from FileLayout."""
        path = snapshot_path / "haloproperties.hdf5"
        layout = discover_file(path)

        # Should be picklable (no live h5py handles)
        pickled = pickle.dumps(layout)
        unpickled = pickle.loads(pickled)

        assert unpickled.error is None
        assert len(unpickled.groups) == len(layout.groups)


class TestDiscoverNestedFile:
    """Test discovery of nested multi-group files."""

    def test_discover_nested_multi_group(self, snapshot_path):
        """Test discovery of haloproperties_multi.hdf5 with nested groups."""
        path = snapshot_path / "haloproperties_multi.hdf5"
        layout = discover_file(path)

        assert layout.error is None
        assert len(layout.groups) == 2, f"Expected 2 groups, got {len(layout.groups)}"

        # Groups should be sorted by path
        paths = [g.path for g in layout.groups]
        assert paths == sorted(paths)

        # Check for distinct header paths
        header_paths = {g.header_path for g in layout.groups}
        assert len(header_paths) == 2, (
            f"Expected 2 distinct header paths, got {header_paths}"
        )

        # header_scopes should yield 2 distinct buckets
        scopes = header_scopes((layout,))
        assert len(scopes) == 2


class TestDiscoverMultiFile:
    """Test serial multi-file discovery."""

    def test_discover_all_serial(self, lightcone_path):
        """Test discover_all on multiple files (serial mode)."""
        paths = [
            lightcone_path / "step_600" / "haloproperties.hdf5",
            lightcone_path / "step_601" / "haloproperties.hdf5",
        ]

        layouts = discover_all(paths, comm=None)

        assert len(layouts) == 2
        assert all(fl.error is None for fl in layouts)

        # Should be sorted by path
        path_strs = [str(fl.path) for fl in layouts]
        assert path_strs == sorted(path_strs)

    def test_discover_all_determinism(self, lightcone_path):
        """Test that discover_all returns same result regardless of input order."""
        paths = [
            lightcone_path / "step_600" / "haloproperties.hdf5",
            lightcone_path / "step_601" / "haloproperties.hdf5",
        ]

        layouts1 = discover_all(paths, comm=None)

        # Shuffle and re-discover
        paths_shuffled = paths.copy()
        random.shuffle(paths_shuffled)
        layouts2 = discover_all(paths_shuffled, comm=None)

        # Should be identical (sorted by path)
        assert len(layouts1) == len(layouts2)
        for fl1, fl2 in zip(layouts1, layouts2):
            assert str(fl1.path) == str(fl2.path)
            assert len(fl1.groups) == len(fl2.groups)
            for g1, g2 in zip(fl1.groups, fl2.groups):
                assert g1.path == g2.path
                assert g1.header_path == g2.header_path


class TestDiscoverMalformed:
    """Test error handling for malformed files."""

    def test_discover_empty_hdf5(self, tmp_path):
        """Test discovery of an empty HDF5 file with no /header."""
        import h5py

        file_path = tmp_path / "empty.hdf5"
        with h5py.File(file_path, "w"):
            # Create an empty file with no header or data
            pass

        layout = discover_file(file_path)

        # Should return error, not raise
        assert layout.error is not None
        assert layout.groups == ()

    def test_discover_non_hdf5(self, tmp_path):
        """Test discovery of a non-HDF5 file."""
        file_path = tmp_path / "not_hdf5.txt"
        file_path.write_text("This is not an HDF5 file")

        layout = discover_file(file_path)

        # Should return error, not raise
        assert layout.error is not None
        assert layout.groups == ()

    def test_discover_header_but_no_data(self, tmp_path):
        """Test discovery of a file with /header but no /data."""
        import h5py

        file_path = tmp_path / "header_only.hdf5"
        with h5py.File(file_path, "w") as f:
            # Create header but no data
            header_grp = f.create_group("header")
            file_grp = header_grp.create_group("file")
            file_grp.attrs["data_type"] = "halo_properties"

        layout = discover_file(file_path)

        # Should return error (no data groups)
        assert layout.error is not None
        assert layout.groups == ()


class TestHelperFunctions:
    """Test helper functions on discovered groups."""

    def test_group_data_type(self, snapshot_path):
        """Test group_data_type helper."""
        path = snapshot_path / "haloproperties.hdf5"
        layout = discover_file(path)
        group = layout.groups[0]

        dtype = group_data_type(group)
        assert dtype == "halo_properties"

    def test_is_particle_group(self, snapshot_path):
        """Test is_particle_group helper."""
        # haloproperties is not a particle group
        path = snapshot_path / "haloproperties.hdf5"
        layout = discover_file(path)
        group = layout.groups[0]

        assert not is_particle_group(group)

        # haloparticles should be a particle group
        path = snapshot_path / "haloparticles.hdf5"
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_particle_group(group)

    def test_is_properties_group(self, snapshot_path):
        """Test is_properties_group helper."""
        # haloproperties should be a properties group
        path = snapshot_path / "haloproperties.hdf5"
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_properties_group(group)

        # galaxyproperties should be a properties group
        path = snapshot_path / "galaxyproperties.hdf5"
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_properties_group(group)

    def test_is_lightcone_group(self, lightcone_path):
        """Test is_lightcone_group helper."""
        path = lightcone_path / "step_600" / "haloproperties.hdf5"
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_lightcone_group(group)

    def test_is_healpix_map_group(self, map_path):
        """Test is_healpix_map_group helper."""
        path = map_path / "test_map.hdf5"
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_healpix_map_group(group)
