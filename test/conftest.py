from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def lightcone_path():
    return Path("test_data") / "lightcone"


@pytest.fixture(scope="session")
def snapshot_path():
    return Path("test_data") / "snapshot"


@pytest.fixture(scope="session")
def diffsky_path():
    return Path("test_data") / "diffsky"
