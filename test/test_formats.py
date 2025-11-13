import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


def test_return_pandas(input_path):
    data = oc.open(input_path).get_data("pandas")
    assert isinstance(data, pd.DataFrame)


def test_return_polars(input_path):
    data = oc.open(input_path).get_data("polars")
    assert isinstance(data, pl.DataFrame)


def test_return_pyarrow(input_path):
    data = oc.open(input_path).get_data("arrow")
    assert isinstance(data, dict)
    assert all(isinstance(v, pa.Array) for v in data.values())


def test_return_pandas_single(input_path):
    dataset = oc.open(input_path)
    column = np.random.choice(dataset.columns)
    data = dataset.select(column).get_data("pandas")

    assert isinstance(data, pd.Series)


def test_return_polars_single(input_path):
    dataset = oc.open(input_path)
    column = np.random.choice(dataset.columns)
    data = dataset.select(column).get_data("polars")
    assert isinstance(data, pl.Series)


def test_return_pyarrow_single(input_path):
    dataset = oc.open(input_path)
    column = np.random.choice(dataset.columns)
    data = dataset.select(column).get_data("arrow")
    assert isinstance(data, pa.Array)
