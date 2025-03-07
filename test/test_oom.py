import pytest
import opencosmo as oc
import numpy as np

@pytest.fixture
def input_path(data_path):
    return data_path / "galaxyproperties.hdf5"


def test_open(input_path):
    read_data = oc.read(input_path).data
    with oc.open(input_path) as f:
        open_data = f.data

    assert np.all(read_data == open_data)
    columns = read_data.columns
    assert all(open_data[col].unit == read_data[col].unit for col in columns)


def test_open_close(input_path):
    with oc.open(input_path) as ds:
        file = ds._Dataset__handler._OutOfMemoryHandler__file
        assert file["data"] is not None

    with pytest.raises(KeyError):
        file["data"]


def test_dataset_close(input_path):
    ds = oc.open(input_path)
    print(type(ds))
    file = ds._Dataset__handler._OutOfMemoryHandler__file
    assert file["data"] is not None
    ds.close()
    with pytest.raises(KeyError):
        file["data"]



