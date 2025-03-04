import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_path():
    try:
        p = os.environ["OPENCOSMO_DATA_PATH"]
    except KeyError:
        pytest.exit("OPENCOSMO_DATA_PATH environment variable not set")
    return Path(p)
