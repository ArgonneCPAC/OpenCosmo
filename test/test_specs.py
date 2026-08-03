from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from opencosmo.io.discover import FileLayout, GroupLayout
from opencosmo.io.specs import (
    DatasetSpec,
    HealpixMapSpec,
    LightconeSpec,
    StructureCollectionSpec,
    group_by_scope,
    match_spec,
)


# Mock header structure to build GroupLayouts without touching HDF5. group_data_type
# does str(header.file.data_type), so a plain str data_type behaves like the real
# DatasetType enum's str().
@dataclass(frozen=True)
class _FakeFile:
    step: int | None
    data_type: str
    is_lightcone: bool


@dataclass(frozen=True)
class _FakeHeader:
    file: _FakeFile


def _group(
    *,
    path: str = "/",
    header_path: str = "/header",
    data_type: str = "halo_properties",
    is_lightcone: bool = False,
    step: int | None = None,
    linked: tuple[str, ...] = (),
    columns: tuple[str, ...] = ("x", "y"),
    dtypes: tuple[str, ...] = ("float64", "float64"),
    has_index: bool = True,
) -> GroupLayout:
    return GroupLayout(
        path=path,
        header_path=header_path,
        header=_FakeHeader(_FakeFile(step, data_type, is_lightcone)),  # type: ignore[arg-type]
        column_names=columns,
        column_dtypes=dtypes,
        row_count=100,
        has_index=has_index,
        linked_target_names=linked,
    )


def _file(path: str, *groups: GroupLayout, error: str | None = None) -> FileLayout:
    return FileLayout(path=Path(path), groups=tuple(groups), error=error)


class TestMatchSpec:
    """match_spec returns the correct spec for each single-scope file type."""

    def test_dataset(self) -> None:
        layouts = (_file("/a.hdf5", _group(data_type="halo_particles")),)
        assert isinstance(match_spec(layouts), DatasetSpec)

    def test_healpix_map(self) -> None:
        layouts = (_file("/m.hdf5", _group(data_type="healpix_map")),)
        assert isinstance(match_spec(layouts), HealpixMapSpec)

    def test_lightcone_single_file(self) -> None:
        layouts = (
            _file(
                "/lc.hdf5",
                _group(data_type="halo_particles", is_lightcone=True, step=0),
            ),
        )
        assert isinstance(match_spec(layouts), LightconeSpec)

    def test_lightcone_multi_file(self) -> None:
        layouts = (
            _file(
                "/lc0.hdf5",
                _group(data_type="halo_particles", is_lightcone=True, step=0),
            ),
            _file(
                "/lc1.hdf5",
                _group(data_type="halo_particles", is_lightcone=True, step=1),
            ),
        )
        assert isinstance(match_spec(layouts), LightconeSpec)

    def test_structure_collection(self) -> None:
        layouts = (
            _file(
                "/sc.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    linked=("haloparticles",),
                ),
                _group(path="/halo_particles", data_type="halo_particles"),
            ),
        )
        assert isinstance(match_spec(layouts), StructureCollectionSpec)

    def test_lone_properties_with_links_is_lightcone_not_sc(self) -> None:
        # Regression: a single properties file carrying /data_linked, opened without
        # its linked children, opens as a Lightcone (old oc.open behavior) — NOT a
        # structure collection. The collection requires >=1 linked child type to be
        # present (i.e. more than one distinct data_type opened together).
        layouts = (
            _file(
                "/haloproperties.hdf5",
                _group(
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
            ),
        )
        assert isinstance(match_spec(layouts), LightconeSpec)

    def test_multiple_properties_of_one_type_is_lightcone_not_sc(self) -> None:
        # Regression: several properties files of a single data_type across redshift
        # steps (each with /data_linked) open as a Lightcone, not a structure
        # collection — there is still only one distinct data_type.
        layouts = (
            _file(
                "/hp0.hdf5",
                _group(
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
            ),
            _file(
                "/hp1.hdf5",
                _group(
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=1,
                ),
            ),
        )
        assert isinstance(match_spec(layouts), LightconeSpec)


class TestPrecedence:
    """Spec ordering resolves the ambiguous cases correctly."""

    def test_lightcone_structure_collection_matches_sc_not_lightcone(self) -> None:
        # A lightcone structure collection has several data_types, so LightconeSpec
        # (single data_type) must not fire; StructureCollectionSpec catches it.
        layouts = (
            _file(
                "/lcsc.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
                _group(
                    path="/halo_particles",
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                ),
            ),
        )
        assert isinstance(match_spec(layouts), StructureCollectionSpec)


class TestGroupByScope:
    """group_by_scope decomposes a multi-scope layout into per-simulation scopes.

    A simulation collection is no longer a spec: the caller splits by header scope
    and builds each scope through the ordinary single-scope path, so each spec only
    ever sees a single scope.
    """

    def test_single_scope_is_one_group(self) -> None:
        layouts = (
            _file(
                "/sc.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    linked=("haloparticles",),
                ),
                _group(path="/halo_particles", data_type="halo_particles"),
            ),
        )
        scopes = group_by_scope(layouts)
        assert list(scopes) == ["/"]
        # The single scope round-trips to its leaf spec.
        assert isinstance(match_spec(scopes["/"]), StructureCollectionSpec)

    def test_nested_scopes_split_and_each_matches_a_leaf(self) -> None:
        layouts = (
            _file(
                "/sim.hdf5",
                _group(
                    path="/scidac1/halo_properties",
                    header_path="/scidac1/header",
                    data_type="halo_properties",
                    linked=("haloparticles",),
                ),
                _group(
                    path="/scidac1/halo_particles",
                    header_path="/scidac1/header",
                    data_type="halo_particles",
                ),
                _group(
                    path="/scidac2/halo_properties",
                    header_path="/scidac2/header",
                    data_type="halo_properties",
                    linked=("haloparticles",),
                ),
                _group(
                    path="/scidac2/halo_particles",
                    header_path="/scidac2/header",
                    data_type="halo_particles",
                ),
            ),
        )
        scopes = group_by_scope(layouts)
        # Sorted per-simulation scopes, each a single-scope sub-layout.
        assert list(scopes) == ["scidac1", "scidac2"]
        for name, sub in scopes.items():
            assert {g.header_path for fl in sub for g in fl.groups} == {
                f"/{name}/header"
            }
            assert isinstance(match_spec(sub), StructureCollectionSpec)

    def test_errored_files_skipped(self) -> None:
        layouts = (
            _file("/good.hdf5", _group(data_type="halo_particles")),
            _file("/bad.hdf5", error="malformed"),
        )
        scopes = group_by_scope(layouts)
        assert list(scopes) == ["/"]
        assert isinstance(match_spec(scopes["/"]), DatasetSpec)


class TestVerify:
    """verify() raises on cross-file structural inconsistency."""

    def test_lightcone_column_mismatch_raises(self) -> None:
        layouts = (
            _file(
                "/lc0.hdf5",
                _group(
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                    columns=("x", "y"),
                ),
            ),
            _file(
                "/lc1.hdf5",
                _group(
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=1,
                    columns=("x", "z"),
                ),
            ),
        )
        spec = match_spec(layouts)
        assert isinstance(spec, LightconeSpec)
        with pytest.raises(ValueError):
            spec.verify(layouts)

    def test_lightcone_dtype_mismatch_raises(self) -> None:
        layouts = (
            _file(
                "/lc0.hdf5",
                _group(
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                    dtypes=("float64", "float64"),
                ),
            ),
            _file(
                "/lc1.hdf5",
                _group(
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=1,
                    dtypes=("float64", "int64"),
                ),
            ),
        )
        with pytest.raises(ValueError):
            LightconeSpec().verify(layouts)

    def test_structure_collection_inconsistent_type_sets_raises(self) -> None:
        # step0 has {halo_properties, halo_particles}, step1 only {halo_properties}.
        layouts = (
            _file(
                "/step0.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
                _group(
                    path="/halo_particles",
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                ),
            ),
            _file(
                "/step1.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=1,
                ),
            ),
        )
        spec = match_spec(layouts)
        assert isinstance(spec, StructureCollectionSpec)
        with pytest.raises(ValueError):
            spec.verify(layouts)

    def test_consistent_structure_collection_passes(self) -> None:
        layouts = (
            _file(
                "/step0.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
                _group(
                    path="/halo_particles",
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                ),
            ),
            _file(
                "/step1.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=1,
                ),
                _group(
                    path="/halo_particles",
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=1,
                ),
            ),
        )
        spec = match_spec(layouts)
        assert isinstance(spec, StructureCollectionSpec)
        # Consistent columns + type sets across steps -> no raise.
        spec.verify(layouts)
