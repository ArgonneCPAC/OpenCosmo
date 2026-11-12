from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class StructurePaths:
    halo_properties: Path
    halo_particles: Path | None = None
    halo_profiles: Path | None = None
    galaxy_properties: Path | None = None
    galaxy_particles: Path | None = None

    @property
    def halos(self) -> list[Path]:
        return [
            path
            for path in (
                self.halo_properties,
                self.halo_particles,
                self.halo_profiles,
            )
            if path is not None
        ]

    @property
    def galaxies(self) -> list[Path]:
        return [
            path
            for path in (self.galaxy_properties, self.galaxy_particles)
            if path is not None
        ]

    @property
    def all(self) -> list[Path]:
        return self.halos + self.galaxies


@dataclass(frozen=True)
class SnapshotPaths:
    root: Path

    @property
    def primary(self) -> StructurePaths:
        root = self.root / "scidac_000"
        return StructurePaths(
            halo_properties=root / "haloproperties.hdf5",
            halo_particles=root / "haloparticles.hdf5",
            halo_profiles=root / "haloprofiles.hdf5",
            galaxy_properties=root / "galaxyproperties.hdf5",
            galaxy_particles=root / "galaxyparticles.hdf5",
        )

    def scidac(self, simulation: int) -> StructurePaths:
        root = self.root / f"scidac_{simulation:03d}"
        return StructurePaths(
            halo_properties=root / "haloproperties.hdf5",
            galaxy_properties=root / "galaxyproperties.hdf5",
        )

    def scidac_linked(self, simulation: int) -> StructurePaths:
        """Properties plus single-row profile links and nested galaxy particles."""
        root = self.root / f"scidac_{simulation:03d}"
        return replace(
            self.scidac(simulation),
            halo_profiles=root / "haloprofiles.hdf5",
            galaxy_particles=root / "galaxyparticles.hdf5",
        )

    @property
    def header(self) -> Path:
        return self.root / "header.hdf5"

    @property
    def multi_simulation(self) -> Path:
        return self.root / "haloproperties_multi.hdf5"

    @property
    def alternate_step(self) -> Path:
        return self.root / "scidac_000" / "haloproperties_step310.hdf5"

    @property
    def mapping_reference(self) -> Path:
        return self.root / "haloproperties_go.hdf5"

    @property
    def halo_mapping(self) -> Path:
        return self.root / "halo_mapping.hdf5"


@dataclass(frozen=True)
class LightconePaths:
    root: Path

    def step(self, step: int) -> StructurePaths:
        root = self.root / f"step_{step}"
        return StructurePaths(
            halo_properties=root / "haloproperties.hdf5",
            halo_particles=root / "haloparticles.hdf5",
            halo_profiles=root / "haloprofiles.hdf5",
            galaxy_properties=root / "galaxyproperties.hdf5",
            galaxy_particles=root / "galaxyparticles.hdf5",
        )


@dataclass(frozen=True)
class DiffskyPaths:
    root: Path

    def core(self, pixel: int) -> Path:
        return self.root / f"lj_{pixel}.hdf5"

    @property
    def invalid(self) -> Path:
        return self.root / "random_data.hdf5"


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path

    @property
    def mass_function(self) -> Path:
        return self.root / "mass_fn.npy"

    @property
    def stacked_profile(self) -> Path:
        return self.root / "stacked_profile.npy"


@dataclass(frozen=True)
class PowerSpectrumPaths:
    root: Path

    @property
    def hydro(self) -> Path:
        return self.root / "hydro.hdf5"

    @property
    def gravity_only(self) -> Path:
        return self.root / "gravity_only.hdf5"


@dataclass(frozen=True)
class TestDataPaths:
    root: Path

    @property
    def snapshot(self) -> SnapshotPaths:
        return SnapshotPaths(self.root / "snapshot")

    @property
    def lightcone(self) -> LightconePaths:
        return LightconePaths(self.root / "lightcone")

    @property
    def diffsky(self) -> DiffskyPaths:
        return DiffskyPaths(self.root / "diffsky")

    @property
    def healpix_map(self) -> Path:
        return self.root / "healpix_map" / "test_map.hdf5"

    @property
    def power_spectrum(self) -> PowerSpectrumPaths:
        return PowerSpectrumPaths(self.root / "power_spectrum")

    @property
    def analysis(self) -> AnalysisPaths:
        return AnalysisPaths(self.root / "analysis")


@pytest.fixture(scope="session")
def test_data() -> TestDataPaths:
    root = os.environ.get("OPENCOSMO_DATA_PATH")
    return TestDataPaths(
        Path(root).expanduser().resolve()
        if root
        else Path(__file__).parents[1] / "test_data"
    )


@pytest.fixture(scope="session", autouse=True)
def _redirect_user_cache_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """
    Point the discovery layout cache at a temp dir for the whole session.

    Without this, every test that opens a file writes to the developer's real
    user cache and leaks state between runs.
    """
    from opencosmo.io import cache

    temp_cache_dir = tmp_path_factory.mktemp("opencosmo-user-cache") / "opencosmo"
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    # Module-level names are not name-mangled, so this patches the real
    # resolver that both the read and write cache-dir paths call.
    original = cache.__user_cache_dir  # type: ignore[attr-defined]
    cache.__user_cache_dir = lambda: temp_cache_dir  # type: ignore[attr-defined]
    try:
        yield
    finally:
        cache.__user_cache_dir = original  # type: ignore[attr-defined]


SNAPSHOT_PARTICLE_FIELDS = {
    "common": {
        "id": "None",
        "mask": "None",
        "mass": "Msun/h",
        "phi_grav": "comoving (km/s)^2",
        "vx": "comoving km/s",
        "vy": "comoving km/s",
        "vz": "comoving km/s",
        "x": "comoving Mpc/h",
        "y": "comoving Mpc/h",
        "z": "comoving Mpc/h",
    },
    "dm_particles": {},
    "gas_particles": {
        "hh": "comoving Mpc/h",
        "mu": "m_hydrogen",
        "rho": "Msun/h / (comoving Mpc/h)^3",
        "sfr": "Msun/yr",
        "uu": "comoving (km/s)^2",
        "wind_decoupling_time": "H0^-1",
        "wind_mass_loading": "None",
        "wind_velocity": "km/s",
        "yhe": "None",
        "zmet": "None",
    },
    "star_particles": {
        "age": "H0^-1",
        "hh": "comoving Mpc/h",
        "initial_star_mass": "Msun/h",
        "yhe": "None",
        "zmet": "None",
    },
    "agn_particles": {
        "age": "H0^-1",
        "bhr": "Msun/yr",
        "ebh": "Msun * (km/s)^2",
        "hh": "comoving Mpc/h",
        "mbh": "Msun/h",
        "rho": "Msun/h / (comoving Mpc/h)^3",
        "uu": "comoving (km/s)^2",
    },
}


@dataclass(frozen=True)
class SnapshotParticleWriter:
    """
    Writes a small snapshot particle file (data type ``snapshot_particles``)
    with the groups, fields, unit strings and octree index of the HACC
    ``snapshotparticles`` product, filled with uniform random particles.
    """

    box_size: float = 64.0
    max_level: int = 2
    redshift: float = 0.5
    h: float = 0.7
    counts: tuple[tuple[str, int], ...] = (
        ("dm_particles", 3000),
        ("gas_particles", 2500),
        ("star_particles", 400),
        ("agn_particles", 200),
    )

    def write(
        self,
        path: Path,
        counts: dict[str, int] | None = None,
        data_type: str = "snapshot_particles",
        extent: dict[str, float] | None = None,
    ) -> Path:
        """``extent`` confines the named types to the box [0, extent)^3."""
        import h5py
        import numpy as np

        rng = np.random.default_rng(42)
        extent = extent or {}
        with h5py.File(path, "w") as f:
            self.__header(data_type).write(f)
            for name, n in (counts or dict(self.counts)).items():
                high = extent.get(name, self.box_size)
                self.__write_type(f.create_group(name), name, n, high, rng)
        return path

    def __header(self, data_type: str):
        from opencosmo.dtypes import FileParameters
        from opencosmo.dtypes.cosmology import CosmologyParameters
        from opencosmo.dtypes.hacc import (
            CosmoToolsParameters,
            HaccSimulationInfo,
            HaccSimulationParameters,
        )
        from opencosmo.header import OpenCosmoHeader

        file_pars = FileParameters.model_validate(
            {
                "data_type": data_type,
                "is_lightcone": False,
                "redshift": self.redshift,
                "step": 400,
            }
        )
        required = {
            "simulation/parameters": HaccSimulationParameters(
                box_size=self.box_size, z_ini=200.0, z_end=0.0, n_steps=625, pm_grid=64
            ),
            "simulation/cosmotools": CosmoToolsParameters(
                cosmotools_steps=frozenset({400}),
                fof_linking_length=0.168,
                fof_pmin=40,
                sod_pmin=500,
                sod_delta_crit=200.0,
                sod_concentration_pmin=500,
                sodbighaloparticles_pmin=5000,
                profiles_nbins=20,
            ),
            "simulation/cosmology": CosmologyParameters.model_validate(
                {
                    "h": self.h,
                    "omega_m": 0.3,
                    "omega_b": 0.05,
                    "omega_l": 0.7,
                    "n_eff_massless": 3.04,
                    "sigma_8": 0.8,
                    "w_0": -1.0,
                    "w_a": 0.0,
                }
            ),
        }
        optional = {"simulation": HaccSimulationInfo(name="snapshot_test")}
        return OpenCosmoHeader(file_pars, required, optional, {})

    def __write_type(self, group, name: str, n: int, high: float, rng) -> None:
        import numpy as np

        positions = rng.uniform(0, high, size=(3, n)).astype(np.float32)
        ids = rng.permutation(10 * n)[:n].astype(np.int64)
        n_cells = 2**self.max_level
        cells = np.clip(
            (positions // (self.box_size / n_cells)).astype(np.int64), 0, n_cells - 1
        )
        # Morton index: bit 3k is bit k of the x cell, 3k+1 of y and 3k+2 of z.
        leaves = np.zeros(n, dtype=np.int64)
        for bit in range(self.max_level):
            for axis in range(3):
                leaves |= ((cells[axis] >> bit) & 1) << (3 * bit + axis)
        order = np.lexsort((ids, leaves))

        fields = SNAPSHOT_PARTICLE_FIELDS["common"] | SNAPSHOT_PARTICLE_FIELDS[name]
        data = group.create_group("data")
        for field, unit in fields.items():
            if field in ("x", "y", "z"):
                values = positions["xyz".index(field)][order]
            elif field == "id":
                values = ids[order]
            elif field == "mask":
                values = np.zeros(n, dtype=np.uint16)
            else:
                values = rng.uniform(1, 2, size=n).astype(np.float32)
            dataset = data.create_dataset(field, data=values)
            dataset.attrs["unit"] = unit
            dataset.attrs["description"] = f"{field} of the particle"

        index = group.create_group("index")
        for level in range(self.max_level + 1):
            blocks = leaves[order] >> (3 * (self.max_level - level))
            size = np.bincount(blocks, minlength=8**level).astype(np.int64)
            level_group = index.create_group(f"level_{level}")
            level_group.create_dataset("start", data=np.cumsum(size) - size)
            level_group.create_dataset("size", data=size)


@pytest.fixture(scope="session")
def snapshot_particle_writer() -> SnapshotParticleWriter:
    return SnapshotParticleWriter()
