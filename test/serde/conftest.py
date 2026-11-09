"""Small cached datasets for serde tests, without simulation files."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from opencosmo.collection.structure import StructureCollection
from opencosmo.collection.structure.state import StructureState
from opencosmo.dataset import Dataset
from opencosmo.dataset.state import DatasetState, state_in_memory
from opencosmo.dtypes.cosmology import CosmologyParameters
from opencosmo.dtypes.file import DatasetType, FileParameters
from opencosmo.header import OpenCosmoHeader
from opencosmo.units import UnitConvention


@pytest.fixture
def cached_structures() -> StructureCollection:
    """Already-aligned hierarchy for column operations; no row-link I/O needed."""
    cosmology = CosmologyParameters(
        h=0.7,
        omega_m=0.3,
        omega_b=0.05,
        omega_l=0.7,
        n_eff_massless=3.046,
        n_eff_massive=0,
        sigma_8=0.8,
        w_0=-1,
        w_a=0,
    )

    def dataset(dtype: str) -> DatasetState:
        header = OpenCosmoHeader(
            FileParameters(
                data_type=DatasetType(dtype),
                is_lightcone=False,
                redshift=0,
                unit_convention=UnitConvention.COMOVING,
            ),
            {"cosmology": cosmology},
            {},
            {},
            UnitConvention.COMOVING,
        )
        return state_in_memory(
            {"mass": np.array([1.0, 2.0, 4.0]) * u.Msun, "tag": np.arange(3)},
            header,
            UnitConvention.COMOVING,
            {},
        )

    galaxies = StructureState(
        dataset("galaxy_properties"),
        {"galaxy_particles": dataset("galaxy_particles")},
        {},
    )
    return StructureCollection(
        StructureState(dataset("halo_properties"), {"galaxies": galaxies}, {})
    )


@pytest.fixture
def cached_dataset(cached_structures: StructureCollection) -> Dataset:
    dataset = cached_structures["halo_properties"]
    assert isinstance(dataset, Dataset)
    return dataset
