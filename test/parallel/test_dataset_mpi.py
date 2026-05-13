import numpy as np
import pytest
from opencosmo.mpi import get_comm_world
from pytest_mpi import parallel_assert

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.mark.parallel(nprocs=4)
def test_take_global(input_path):
    comm = get_comm_world()

    ds = oc.open(input_path)
    total_length = sum(comm.allgather(len(ds)))
    n_to_take = np.random.randint(total_length // 4, int(total_length * 0.75))
    n_to_take = comm.bcast(n_to_take)

    ds = ds.take(n_to_take, mode="global")
    all_lengths = comm.allgather(len(ds))

    parallel_assert(sum(all_lengths) == n_to_take)


# ── take_range global, unsorted ──────────────────────────────────────────────


@pytest.mark.parallel(nprocs=4)
def test_take_range_global_start(input_path):
    """First n global rows land on the correct ranks with the right counts."""
    comm = get_comm_world()
    ds = oc.open(input_path)

    lengths = np.array(comm.allgather(len(ds)), dtype=np.int64)
    total = int(np.sum(lengths))
    n = total // 3

    ds_taken = ds.take_range(0, n, mode="global")

    rank = comm.Get_rank()
    offset = int(np.sum(lengths[:rank]))
    expected_local = max(0, min(int(lengths[rank]), n - offset))

    parallel_assert(
        len(ds_taken) == expected_local,
        f"rank {rank}: expected {expected_local} rows, got {len(ds_taken)}",
    )
    parallel_assert(sum(comm.allgather(len(ds_taken))) == n)


@pytest.mark.parallel(nprocs=4)
def test_take_range_global_end(input_path):
    """Last n global rows land on the correct ranks with the right counts."""
    comm = get_comm_world()
    ds = oc.open(input_path)

    lengths = np.array(comm.allgather(len(ds)), dtype=np.int64)
    total = int(np.sum(lengths))
    n = total // 3
    global_start = total - n

    ds_taken = ds.take_range(global_start, total, mode="global")

    rank = comm.Get_rank()
    offset = int(np.sum(lengths[:rank]))
    expected_local = max(
        0,
        min(int(lengths[rank]), total - offset) - max(0, global_start - offset),
    )

    parallel_assert(
        len(ds_taken) == expected_local,
        f"rank {rank}: expected {expected_local} rows, got {len(ds_taken)}",
    )
    parallel_assert(sum(comm.allgather(len(ds_taken))) == n)


# ── take_range global, sorted ─────────────────────────────────────────────────
#
# The sorted tests verify value-level correctness: after a global range take on
# a sorted dataset, every selected value must satisfy the global threshold
# implied by the range position.  We derive the expected threshold by gathering
# all values from all ranks before the take, then checking the invariant after.


@pytest.mark.parallel(nprocs=4)
def test_take_range_global_sorted_start(input_path):
    """Global start on sorted data selects the n globally smallest values."""
    comm = get_comm_world()
    ds = oc.open(input_path).sort_by("fof_halo_mass")

    total = sum(comm.allgather(len(ds)))
    n = total // 3

    # Gather original values to compute the expected threshold before the take.
    original = ds.select("fof_halo_mass").get_data("numpy")
    all_original = np.concatenate(comm.allgather(original))
    threshold = np.sort(all_original)[n - 1]

    ds_taken = ds.take_range(0, n, mode="global")

    selected = ds_taken.select("fof_halo_mass").get_data("numpy")
    all_selected = np.concatenate(comm.allgather(selected))

    parallel_assert(len(all_selected) == n)
    parallel_assert(
        np.all(all_selected <= threshold),
        "some selected values exceed the global n-th smallest threshold",
    )


@pytest.mark.parallel(nprocs=4)
def test_take_range_global_sorted_end(input_path):
    """Global end on sorted data selects the n globally largest values."""
    comm = get_comm_world()
    ds = oc.open(input_path).sort_by("fof_halo_mass")

    total = sum(comm.allgather(len(ds)))
    n = total // 3

    original = ds.select("fof_halo_mass").get_data("numpy")
    all_original = np.concatenate(comm.allgather(original))
    threshold = np.sort(all_original)[::-1][n - 1]

    ds_taken = ds.take_range(total - n, total, mode="global")

    selected = ds_taken.select("fof_halo_mass").get_data("numpy")
    all_selected = np.concatenate(comm.allgather(selected))

    parallel_assert(len(all_selected) == n)
    parallel_assert(
        np.all(all_selected >= threshold),
        "some selected values fall below the global n-th largest threshold",
    )


# ── take global end ───────────────────────────────────────────────────────────


@pytest.mark.parallel(nprocs=4)
def test_take_global_end(input_path):
    """take(n, at='end', mode='global') selects the last n rows across all ranks."""
    comm = get_comm_world()
    ds = oc.open(input_path)

    lengths = np.array(comm.allgather(len(ds)), dtype=np.int64)
    total = int(np.sum(lengths))
    n = total // 3
    global_start = total - n

    ds_taken = ds.take(n, at="end", mode="global")

    rank = comm.Get_rank()
    offset = int(np.sum(lengths[:rank]))
    expected_local = max(
        0,
        min(int(lengths[rank]), total - offset) - max(0, global_start - offset),
    )

    parallel_assert(
        len(ds_taken) == expected_local,
        f"rank {rank}: expected {expected_local} rows, got {len(ds_taken)}",
    )
    parallel_assert(sum(comm.allgather(len(ds_taken))) == n)


@pytest.mark.parallel(nprocs=4)
def test_take_global_end_sorted(input_path):
    """take(n, at='end', mode='global') on sorted data selects the n globally largest values."""
    comm = get_comm_world()
    ds = oc.open(input_path).sort_by("fof_halo_mass")

    total = sum(comm.allgather(len(ds)))
    n = total // 3

    original = ds.select("fof_halo_mass").get_data("numpy")
    all_original = np.concatenate(comm.allgather(original))
    threshold = np.sort(all_original)[::-1][n - 1]

    ds_taken = ds.take(n, at="end", mode="global")

    selected = ds_taken.select("fof_halo_mass").get_data("numpy")
    all_selected = np.concatenate(comm.allgather(selected))

    parallel_assert(len(all_selected) == n)
    parallel_assert(
        np.all(all_selected >= threshold),
        "some selected values fall below the global n-th largest threshold",
    )


# ── take_range global, middle ─────────────────────────────────────────────────


@pytest.mark.parallel(nprocs=4)
def test_take_range_global_middle(input_path):
    """A middle window of global rows lands on the correct ranks with the right counts."""
    comm = get_comm_world()
    ds = oc.open(input_path)

    lengths = np.array(comm.allgather(len(ds)), dtype=np.int64)
    total = int(np.sum(lengths))
    global_start = total // 4
    global_end = 3 * total // 4

    ds_taken = ds.take_range(global_start, global_end, mode="global")

    rank = comm.Get_rank()
    offset = int(np.sum(lengths[:rank]))
    expected_local = max(
        0,
        min(int(lengths[rank]), global_end - offset) - max(0, global_start - offset),
    )

    parallel_assert(
        len(ds_taken) == expected_local,
        f"rank {rank}: expected {expected_local} rows, got {len(ds_taken)}",
    )
    parallel_assert(sum(comm.allgather(len(ds_taken))) == global_end - global_start)


@pytest.mark.parallel(nprocs=4)
def test_take_range_global_sorted_middle(input_path):
    """A middle window on sorted data selects the correct globally-ranked values."""
    comm = get_comm_world()
    ds = oc.open(input_path).sort_by("fof_halo_mass")

    total = sum(comm.allgather(len(ds)))
    global_start = total // 4
    global_end = 3 * total // 4
    size = global_end - global_start

    original = ds.select("fof_halo_mass").get_data("numpy")
    all_original = np.concatenate(comm.allgather(original))
    sorted_all = np.sort(all_original)
    lower_threshold = sorted_all[global_start]
    upper_threshold = sorted_all[global_end - 1]

    ds_taken = ds.take_range(global_start, global_end, mode="global")

    selected = ds_taken.select("fof_halo_mass").get_data("numpy")
    all_selected = np.concatenate(comm.allgather(selected))

    parallel_assert(len(all_selected) == size)
    parallel_assert(
        np.all(all_selected >= lower_threshold),
        "some selected values fall below the lower global threshold",
    )
    parallel_assert(
        np.all(all_selected <= upper_threshold),
        "some selected values exceed the upper global threshold",
    )
