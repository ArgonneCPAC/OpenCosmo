from pathlib import Path

import numpy as np
import pytest
from pyarrow import parquet as pq

import opencosmo as oc
from opencosmo.io import write_parquet


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def halo_structure_paths(snapshot_path: Path):
    files = ["haloparticles.hdf5", "haloproperties.hdf5", "sodproperties.hdf5"]
    hdf_files = [snapshot_path / file for file in files]
    return list(hdf_files)


@pytest.fixture
def haloproperties_600_path(lightcone_path):
    return lightcone_path / "step_600" / "haloproperties.hdf5"


@pytest.fixture
def haloproperties_601_path(lightcone_path):
    return lightcone_path / "step_601" / "haloproperties.hdf5"


def test_dump_dataset(input_path, tmp_path):
    dataset = oc.open(input_path)
    write_parquet(tmp_path / "test.parquet", dataset)
    table = pq.read_table(tmp_path / "test.parquet")
    data = dataset.get_data("numpy")

    for col in dataset.columns:
        assert np.all(data[col] == table[col])


def test_dump_dataset_unit_transition(input_path, tmp_path):
    dataset = oc.open(input_path).with_units("physical")
    write_parquet(tmp_path / "test.parquet", dataset)
    table = pq.read_table(tmp_path / "test.parquet")
    data = dataset.get_data("numpy")

    for col in dataset.columns:
        assert np.all(data[col] == table[col])


def test_dump_lightcone(haloproperties_600_path, haloproperties_601_path, tmp_path):
    dataset = oc.open(haloproperties_600_path, haloproperties_601_path)
    write_parquet(tmp_path / "test.parquet", dataset)
    table = pq.read_table(tmp_path / "test.parquet")
    data = dataset.get_data("numpy")

    for col in dataset.columns:
        coldata = data[col]
        if coldata.dtype.names is not None:
            continue

        assert np.all(data[col] == table[col])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dump_structure(halo_structure_paths, tmp_path):
    dataset = (
        oc.open(*halo_structure_paths).filter(oc.col("fof_halo_mass") > 1e14).take(1)
    )
    write_parquet(tmp_path, dataset)
    _ = list(tmp_path.glob("*.parquet"))
    assert False
