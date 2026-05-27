from __future__ import annotations

from functools import reduce, wraps
from typing import TYPE_CHECKING, Callable, Iterable

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


def offset_3d(
    coord_name_a: str, coord_name_b: str, labels: Iterable[str] = ["x", "y", "z"]
):
    """
    Create a derived column that contains the magnitude of the offset between two sets of 3d coordinates.
    For exmaple, to get the magnitude of the difference between the FoF halo centers and the SOD halo centers:

    .. code-block:: python

        from opencosmo.column import offset_3d
        import opencosmo as oc

        dataset = oc.open("haloproperties.hdf5")

        offset_column = offset_3d("fof_halo_com", "sod_halo_com")
        dataset = dataset.with_new_columns(offset=offset_column)


    This function assumes that the columns are named "fof_halo_center_{x, y, z}" and "sod_halo_center_{x, y, z}",
    you can choose different labels by setting the :code:`labels` argument.

    This function outputs a derived column that can be passed into :py:meth:`with_new_columns <opencosmo.Dataset.with_new_files>. This function will never fail, but :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`
    will if the columns do not all have the same units.

    Parameters
    ----------
    coord_name_a: str
        The base name of the first coordinate

    coord_name_b: str
        The base name of the second coordinate

    labels: Iterable[str], default = ["x", "y", "z"]
        The coordinate labels. The names of the columns are assumed to be "{coord_name_a}_{labels}" and
        "{coord_name_b}_{labels}"

    """
    delta_coords = tuple(
        map(
            lambda label: col(f"{coord_name_a}_{label}")
            - col(f"{coord_name_b}_{label}"),
            labels,
        )
    )
    return norm_cols(*delta_coords)


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

    Parameters
    ----------
    *magnitudes: str | Column | DerivedColumn
        Any number of magnitude columns. You can pass in simple column names, columns constructred
        with :py:meth:`opencosmo.col`, or columns created from combinations of other columns

    Returns
    -------
    new_column: DerivedColumn
        A new derived column that can be passed into :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`
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
    Get the euclidian norm of any number of columns. This function takes in the names
    of the magnitude columns, and produces a DerivedColumn that can be passed into
    :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`

    This function will never fail, but :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`
    will if the columns do not have the same units.

    Parameters
    ----------
    *columns: str | Column | DerivedColumn
        Any number of columns. You can pass in simple column names, columns constructred
        with :py:meth:`opencosmo.col`, or columns created from combinations of other columns

    Returns
    -------
    new_column: DerivedColumn
        A new derived column that can be passed into :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`
    """
    if len(columns) < 2:
        raise ValueError("Expected at least two magnitudes to add together!")

    squared_columns = map(lambda column: column**2, columns)
    sum_squared = reduce(lambda acc, col_sq: acc + col_sq, squared_columns)
    return sum_squared.sqrt()
