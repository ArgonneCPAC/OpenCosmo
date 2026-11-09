"""Result messages: summaries of resultant datasets and collections."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import pytest
from opencosmo.collection.lightcone import Lightcone
from opencosmo.collection.lightcone import state as lcst
from opencosmo.dataset.state import state_in_memory
from opencosmo.dtypes.cosmology import CosmologyParameters
from opencosmo.dtypes.file import DatasetType, FileParameters
from opencosmo.dtypes.lightcone import LightconeParams
from opencosmo.header import OpenCosmoHeader
from opencosmo.serde import (
    DatasetSummary,
    LightconeSummary,
    ResultMessage,
    StructureCollectionSummary,
    summarize,
)
from opencosmo.units import UnitConvention
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from opencosmo.collection.structure import StructureCollection
    from opencosmo.dataset import Dataset

ADAPTER: TypeAdapter[ResultMessage] = TypeAdapter(ResultMessage)


def _lightcone_leaf(redshift: float):
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
    header = OpenCosmoHeader(
        FileParameters(
            data_type=DatasetType("halo_properties"),
            is_lightcone=True,
            redshift=redshift,
            unit_convention=UnitConvention.COMOVING,
        ),
        {"cosmology": cosmology},
        {},
        {"lightcone": LightconeParams(z_range=(redshift, redshift))},
        UnitConvention.COMOVING,
    )
    return state_in_memory(
        {"mass": np.array([1.0, 2.0]) * u.Msun},
        header,
        UnitConvention.COMOVING,
        {},
    )


def test_dataset_summary(cached_dataset: Dataset):
    summary = summarize(cached_dataset)
    assert isinstance(summary, DatasetSummary)
    assert summary.kind == "dataset"
    assert summary.length == len(cached_dataset)
    assert [c.name for c in summary.columns] == cached_dataset.columns
    mass = next(c for c in summary.columns if c.name == "mass")
    assert mass.unit == "solMass"
    assert mass.description == cached_dataset.descriptions["mass"]


def test_structure_collection_summary(cached_structures: StructureCollection):
    summary = summarize(cached_structures)
    assert isinstance(summary, StructureCollectionSummary)
    assert summary.kind == "structure_collection"
    assert summary.length == len(cached_structures)
    assert summary.dtype == cached_structures.dtype
    assert set(summary.datasets) == set(cached_structures.keys())
    galaxies = summary.datasets["galaxies"]
    assert isinstance(galaxies, StructureCollectionSummary)
    particles = galaxies.datasets["galaxy_particles"]
    assert isinstance(particles, DatasetSummary)
    assert particles.length == 3


def test_lightcone_summary():
    lightcone = Lightcone(
        lcst.from_leaves(
            (((600,), _lightcone_leaf(0.0)), ((601,), _lightcone_leaf(0.1))),
            z_range=(0.0, 0.1),
        )
    )
    summary = summarize(lightcone)
    assert isinstance(summary, LightconeSummary)
    assert summary.kind == "lightcone"
    assert summary.length == 4
    assert set(summary.datasets) == {"600", "601"}
    assert all(isinstance(child, DatasetSummary) for child in summary.datasets.values())


def test_result_message_round_trips_json(cached_structures: StructureCollection):
    payload = json.loads(summarize(cached_structures).model_dump_json())
    assert ADAPTER.validate_python(payload) == summarize(cached_structures)


def test_summarize_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Cannot summarize"):
        summarize(object())
