from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from mpi4py import MPI

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture
def per_test_dir(request: pytest.FixtureRequest):
    """
    A unique, MPI-shared scratch directory for a single test.

    Rank 0 allocates the directory with ``tempfile.mkdtemp`` and broadcasts it;
    every rank reads/writes the same path (they share one filesystem). Only rank
    0 owns the directory's lifecycle, and removal is gated behind a barrier so a
    lagging rank can never lose a file another rank just wrote.

    We deliberately do NOT use pytest's ``tmp_path_factory`` here. Under
    ``mpiexec`` each rank is a separate pytest process; each computes its own
    ``pytest-of-<user>/pytest-N`` basetemp and rotates it with a keep-last-3
    retention policy. Because the ranks land on different ``N`` and rotate a
    shared parent directory independently, one rank's cleanup can delete a
    ``pytest-N`` tree another rank is actively writing into — an intermittent
    ``FileNotFoundError`` seen only in CI, where the temp root is reused across
    runs. ``mkdtemp`` sidesteps that machinery entirely.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # request.node.nodeid is unique across parameterizations; sanitize for
        # the filesystem so the scratch dir is easy to attribute when debugging.
        nodeid = (
            request.node.nodeid.replace("/", "_")
            .replace("::", "__")
            .replace("[", "_")
            .replace("]", "_")
        )
        path: Path | None = Path(tempfile.mkdtemp(prefix=f"{nodeid}__"))
    else:
        path = None

    path = comm.bcast(path, root=0)

    try:
        yield path
    finally:
        # Every rank must be done with the directory before rank 0 removes it.
        comm.Barrier()
        if rank == 0 and IN_GITHUB_ACTIONS:
            shutil.rmtree(path, ignore_errors=True)
