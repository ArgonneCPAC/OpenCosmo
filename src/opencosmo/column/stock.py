from __future__ import annotations

from functools import reduce, wraps
from typing import TYPE_CHECKING, Callable

import astropy.units as u

if TYPE_CHECKING:
    from .column import Column, DerivedColumn

from .column import col


def into_cols(func: Callable):
    @wraps(func)
    def wrapper(*columns: str | Column | DerivedColumn, **kwargs):
        new_columns = tuple(
            map(
                lambda colname: col(colname) if isinstance(colname, str) else colname,
                columns,
            )
        )
        return func(*new_columns, **kwargs)

    return wrapper


@into_cols
def add_mag_cols(*magnitudes: Column | DerivedColumn):
    """
    Add together any number of magnitude columns to get a total magnitude. This function
    takes in the names of the magnitude columns, and produces a DerivedColumn that can be
    passed into :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`

    This function will never fail, but :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`
    will if you include columns that are not magnitudes.

    .. code-block:: python

        import opencosmo as oc
        from opencosmo.column import add_mag_cols

        dataset = oc.open("catalog.hdf5")
        mag_total = add_mag_cols("mag_g", "mag_r", "mag_i", "mag_z", "mag_y")

        dataset = dataset.with_new_columns(mag_total=mag_total)


    """
    if len(magnitudes) < 2:
        raise ValueError("Expected at least two magnitudes to add together!")
    fluxes = map(
        lambda column: (-0.4 * column).exp10(expected_unit_container=u.MagUnit),
        magnitudes,
    )
    total_flux = next(fluxes)
    for flux in fluxes:
        total_flux += flux

    return -2.5 * total_flux.log10(u.MagUnit)


@into_cols
def norm_cols(*columns: Column | DerivedColumn):
    """
    Get the

    """
    if len(columns) < 2:
        raise ValueError("Expected at least two magnitudes to add together!")

    squared_columns = map(lambda column: column**2, columns)
    sum_squared = reduce(lambda acc, col_sq: acc + col_sq, squared_columns)
    return sum_squared.sqrt()
