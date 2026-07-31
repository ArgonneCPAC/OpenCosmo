"""Unit tests for the redshift-distribution planner's contiguous partitioner.

``partition_contiguous`` is a pure function, so it is tested here serially rather
than under MPI. The MPI-level behavior (each rank receiving one contiguous
redshift range) is exercised in test/parallel/test_lc_mpi.py.
"""

import itertools

import pytest
from opencosmo.collection.lightcone.distribute import partition_contiguous


def _is_contiguous_cover(groups, n):
    """Every group is an ascending run, and concatenation reproduces range(n)."""
    flat = [i for group in groups for i in group]
    if flat != list(range(n)):
        return False
    for group in groups:
        if group and group != list(range(group[0], group[-1] + 1)):
            return False
    return True


def _optimal_max_sum(weights, k):
    """Brute-force minimal achievable max-group-sum over all contiguous splits."""
    n = len(weights)
    if n == 0:
        return 0
    k = min(k, n)
    best = None
    # Choose k-1 cut points among the n-1 internal boundaries.
    for cuts in itertools.combinations(range(1, n), k - 1):
        bounds = [0, *cuts, n]
        sums = [sum(weights[bounds[i] : bounds[i + 1]]) for i in range(len(bounds) - 1)]
        m = max(sums)
        if best is None or m < best:
            best = m
    return best


@pytest.mark.parametrize(
    "weights,k",
    [
        ([100, 100, 100, 100, 100, 100], 3),
        ([10, 20, 40, 80, 160, 320], 4),  # skewed, low-z steps are small
        ([320, 160, 80, 40, 20, 10], 4),  # skewed the other way
        ([1, 1, 1, 1, 1], 2),
        ([5, 1, 5, 1, 5, 1, 5], 3),
        ([7], 1),
    ],
)
def test_partition_is_contiguous_and_optimal(weights, k):
    groups = partition_contiguous(weights, k)
    assert len(groups) == k
    assert _is_contiguous_cover(groups, len(weights))

    sums = [sum(weights[i] for i in group) for group in groups]
    assert max(sums) == _optimal_max_sum(weights, k)


def test_partition_fewer_items_than_groups():
    # 2 items, 4 groups: first two groups get one item each, the rest are empty.
    groups = partition_contiguous([115723, 100885], 4)
    assert groups == [[0], [1], [], []]


def test_partition_single_group():
    groups = partition_contiguous([3, 1, 4, 1, 5], 1)
    assert groups == [[0, 1, 2, 3, 4]]


def test_partition_empty_weights():
    groups = partition_contiguous([], 3)
    assert groups == [[], [], []]


def test_partition_rejects_nonpositive_groups():
    with pytest.raises(ValueError):
        partition_contiguous([1, 2, 3], 0)
