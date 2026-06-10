"""MPI tests for Lightcone-scoped derived scalars.

The scope materializes scoped columns at the lightcone level on each rank,
which means the per-rank vstack of children is reduced by the rank-level
Reducer (MpiReducer when MPI is active). Verifying these tests under MPI is
the only way to catch double-counting or cross-rank communication failures.
"""

import numpy as np
import pytest
from mpi4py import MPI

import opencosmo as oc


@pytest.fixture
def haloproperties_600_path(lightcone_path):
    return lightcone_path / "step_600" / "haloproperties.hdf5"


@pytest.fixture
def haloproperties_601_path(lightcone_path):
    return lightcone_path / "step_601" / "haloproperties.hdf5"


def _global_raw(lc):
    """Gather the per-rank vstacked fof_halo_mass across all ranks."""
    comm = MPI.COMM_WORLD
    local = np.concatenate(
        [
            np.asarray(child.select("fof_halo_mass").get_data("numpy"))
            for child in lc.values()
        ]
    )
    return np.concatenate(comm.allgather(local))


@pytest.mark.parallel(nprocs=4)
def test_lc_scope_mpi_global_zscore(haloproperties_600_path, haloproperties_601_path):
    """The reducer must combine across both children AND MPI ranks."""
    comm = MPI.COMM_WORLD
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    m = oc.col("fof_halo_mass")
    data = lc.with_new_columns(zscore=(m - m.mean()) / m.std()).get_data()

    local_zscore = np.asarray(data["zscore"])
    all_zscore = np.concatenate(comm.allgather(local_zscore))
    assert np.isclose(all_zscore.mean(), 0.0, atol=1e-6)
    assert np.isclose(all_zscore.std(), 1.0, atol=1e-6)


@pytest.mark.parallel(nprocs=4)
def test_lc_scope_mpi_scalar_select_is_global(
    haloproperties_600_path, haloproperties_601_path
):
    """A scope-only scalar select must return the global min across all ranks."""
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    raw = _global_raw(lc)

    result = lc.select(min_mass=oc.col("fof_halo_mass").min()).get_data()
    arr = np.asarray(result)
    # On each rank the value should equal the global min (broadcast or scalar).
    assert np.all(np.isclose(arr, float(np.min(raw))))


@pytest.mark.parallel(nprocs=4)
def test_lc_scope_mpi_filter_against_global_mean(
    haloproperties_600_path, haloproperties_601_path
):
    """Filter threshold must be the lightcone-wide AND rank-wide mean."""
    comm = MPI.COMM_WORLD
    lc = oc.open(haloproperties_600_path, haloproperties_601_path)
    m = oc.col("fof_halo_mass")
    raw = _global_raw(lc)
    threshold = float(np.mean(raw))

    filtered = lc.filter(m > m.mean())
    local_kept = np.asarray(filtered.select("fof_halo_mass").get_data("numpy"))
    all_kept = np.concatenate(comm.allgather(local_kept))

    expected = raw[raw > threshold]
    assert len(all_kept) == len(expected)
    assert np.isclose(np.sort(all_kept), np.sort(expected)).all()
