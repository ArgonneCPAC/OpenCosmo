from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Ordered by priority
KNOWN_SCALE_FACTOR_COLUMNS = ("fof_halo_center_a", "gal_center_a")
KNOWN_REDSHIFT_COLUMNS = ("redshift", "redshift_true")


def get_scale_factor_column(columns: Iterable[str]) -> str | None:
    """Return the highest-priority known scale factor column, if any."""
    return __first_known(KNOWN_SCALE_FACTOR_COLUMNS, columns)


def get_redshift_column(columns: Iterable[str]) -> str | None:
    """Return the highest-priority known redshift column, if any."""
    return __first_known(KNOWN_REDSHIFT_COLUMNS, columns)


def __first_known(known: tuple[str, ...], columns: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in known if name in available), None)
