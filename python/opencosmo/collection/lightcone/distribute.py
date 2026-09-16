from __future__ import annotations


def partition_contiguous(weights: list[int], k: int) -> list[list[int]]:
    """
    Split a sequence of ``weights`` into ``k`` *contiguous* groups, minimizing the
    largest group sum (the classic linear-partition problem, solved optimally with
    dynamic programming).

    This is what makes each rank receive one contiguous run of redshift steps of
    roughly equal data volume, rather than an arbitrary scatter of steps that
    merely happen to balance the row counts.

    Parameters
    ----------
    weights : list[int]
        Per-item weights, in the order the items must stay in (redshift order).
    k : int
        Number of contiguous groups (ranks) to produce.

    Returns
    -------
    list[list[int]]
        ``k`` lists of item indices. Each inner list is a contiguous, ascending
        run of indices; concatenating them in order reproduces ``range(len(weights))``.
        When there are fewer items than groups, the surplus trailing groups are empty.
    """
    if k <= 0:
        raise ValueError("Number of groups must be positive")

    n = len(weights)
    if n == 0:
        return [[] for _ in range(k)]
    if k >= n:
        # One item per group for the first n groups; the rest are empty.
        return [[i] for i in range(n)] + [[] for _ in range(k - n)]

    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + weights[i]

    # dp[j][i] = minimal achievable max-group-sum when splitting the first i items
    # into j contiguous groups. split[j][i] records where the last group starts.
    inf = float("inf")
    dp: list[list[float]] = [[inf] * (n + 1) for _ in range(k + 1)]
    split: list[list[int]] = [[0] * (n + 1) for _ in range(k + 1)]
    for i in range(1, n + 1):
        dp[1][i] = prefix[i]
    for j in range(2, k + 1):
        # Need at least j items to form j non-empty groups.
        for i in range(j, n + 1):
            best = inf
            best_m = j - 1
            for m in range(j - 1, i):  # last group covers items [m, i)
                candidate = max(dp[j - 1][m], prefix[i] - prefix[m])
                if candidate < best:
                    best = candidate
                    best_m = m
            dp[j][i] = best
            split[j][i] = best_m

    groups: list[list[int]] = []
    i = n
    for j in range(k, 0, -1):
        m = split[j][i] if j > 1 else 0
        groups.append(list(range(m, i)))
        i = m
    groups.reverse()
    return groups
