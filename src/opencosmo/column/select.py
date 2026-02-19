""" """

import re
from typing import Iterable


def get_column_selection(
    select_from: Iterable[str], select_by: Iterable[str]
) -> tuple[set[str], set[str]]:
    """
    Selects a list of columns from another list of columns. Supports wildcards.

    Returns two sets. The first is the set of matched columns. The second is the set
    of selections that could not be matched.
    """

    select_from = set(select_from)
    select_by = set(select_by)

    wildcards = set(n for n in select_by if "*" in n)
    complete = select_by - wildcards

    complete_matches = complete.intersection(select_from)
    complete_missing = complete.difference(select_from)

    if not any(wildcards):
        return complete_matches, complete_missing

    wildcard_matches = __evaluate_wildcards(select_from, wildcards)
    return complete_matches.union(wildcard_matches), complete_missing


def __evaluate_wildcards(select_from: set[str], wildcards: Iterable[str]):
    wildcards = list(map(lambda w: w.replace("*", ".*"), wildcards))
    regex = re.compile("|".join(wildcards))
    matches = set(filter(lambda n: re.fullmatch(regex, n), select_from))
    return matches
