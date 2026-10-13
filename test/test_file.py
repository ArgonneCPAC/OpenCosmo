import pytest

from opencosmo.cosmology import read_cosmology


def test_reader_failer():
    with pytest.raises(FileNotFoundError):
        read_cosmology("non_existent_file.hdf5")
