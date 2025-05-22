import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def lightcone_path():
    try:
        p = os.environ["OPENCOSMO_DATA_PATH"]
    except KeyError:
        pytest.exit("OPENCOSMO_DATA_PATH environment variable not set")
    return Path(p / "lightcone")


@pytest.fixture(scope="session")
def snapshot_path():
    try:
        p = os.environ["OPENCOSMO_DATA_PATH"]
    except KeyError:
        pytest.exit("OPENCOSMO_DATA_PATH environment variable not set")
    return Path(p) / "snapshot"
