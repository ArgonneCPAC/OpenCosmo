from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from os import PathLike

    import astropy.units as u
    from astropy.io import fits

__all__ = ["write_cutout"]

CENTER_EXTNAME = "CENTER"
_MAX_FITS_NAME_LENGTH = 68


def write_cutout(
    path: str | PathLike[str],
    to_write: fits.HDUList | tuple[Mapping[str, object], fits.HDUList],
    overwrite: bool = False,
) -> None:
    """
    Write a FITS cutout to disk.

    ``to_write`` may be either a bare :py:class:`astropy.io.fits.HDUList`, which
    is written as-is, or a ``(row, hdulist)`` tuple as produced by
    :py:meth:`opencosmo.Lightcone.cutouts`. In the latter case the catalog row
    is stored as a one-row binary table extension named ``CENTER``, with units
    preserved and scalar and vector values both supported. The original
    ``HDUList`` is never mutated or closed.

    Parameters
    ----------
    path : str or PathLike
        The file to write to.
    to_write : astropy.io.fits.HDUList or tuple of (Mapping, astropy.io.fits.HDUList)
        The cutout to write. A ``(row, hdulist)`` tuple adds a ``CENTER``
        metadata extension built from ``row``.
    overwrite : bool, default = False
        If true, an existing file at ``path`` is overwritten. Otherwise a
        :py:exc:`FileExistsError` is raised before any output file is created.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is false.
    TypeError
        If ``to_write`` is not an ``HDUList`` or ``(row, HDUList)`` tuple, or a
        metadata value has an unsupported type or object/structured dtype.
    ValueError
        If the metadata row is empty, a column name is invalid, the tuple does
        not have exactly two elements, or the ``HDUList`` already contains a
        ``CENTER`` extension.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)

    from astropy.io import fits

    center_hdu = None
    if isinstance(to_write, tuple):
        if len(to_write) != 2:
            raise ValueError(
                "to_write tuple must be exactly (row, hdulist), "
                f"got {len(to_write)} elements"
            )
        row, hdulist = to_write
        if not isinstance(row, Mapping):
            raise TypeError(
                "The first element of a to_write tuple must be a mapping of "
                f"metadata, got {type(row).__name__}"
            )
        if not isinstance(hdulist, fits.HDUList):
            raise TypeError(
                "The second element of a to_write tuple must be an "
                f"astropy.io.fits.HDUList, got {type(hdulist).__name__}"
            )
        center_hdu = __make_center_hdu(row)
    elif isinstance(to_write, fits.HDUList):
        hdulist = to_write
    else:
        raise TypeError(
            "to_write must be an astropy.io.fits.HDUList or a (row, "
            f"astropy.io.fits.HDUList) tuple, got {type(to_write).__name__}"
        )

    # FITS writing may update headers and internal HDU state.
    output = fits.HDUList([hdu.copy() for hdu in hdulist])
    if center_hdu is not None:
        if any(hdu.name.upper() == CENTER_EXTNAME for hdu in output):
            raise ValueError(
                f"Cannot add a {CENTER_EXTNAME} metadata extension: the HDUList "
                f"already contains an extension named {CENTER_EXTNAME}"
            )
        output.append(center_hdu)
    output.writeto(path, overwrite=overwrite)


def __make_center_hdu(row: Mapping[str, object]) -> fits.BinTableHDU:
    from astropy.io import fits
    from astropy.table import Table

    if len(row) == 0:
        raise ValueError("Cannot write a cutout with an empty metadata row")

    names = list(row.keys())
    for name in names:
        __validate_name(name)

    table = Table()
    for name in sorted(names):
        value = __validate_value(name, row[name])
        table[name] = np.expand_dims(value, 0)
    return fits.BinTableHDU(table, name=CENTER_EXTNAME)


def __validate_name(name: object) -> None:
    if not isinstance(name, str):
        raise TypeError(
            f"Metadata column names must be strings, got {type(name).__name__}"
        )
    if not name.strip():
        raise ValueError("Metadata column names must be non-empty")
    if len(name) > _MAX_FITS_NAME_LENGTH:
        raise ValueError(
            f"Metadata column name {name!r} exceeds the "
            f"{_MAX_FITS_NAME_LENGTH}-character FITS limit"
        )


def __validate_value(name: str, value: object) -> np.ndarray | u.Quantity:
    import astropy.units as u

    if isinstance(value, u.Quantity):
        array = np.asarray(value.value)
    elif isinstance(value, (str, bool, np.bool_, Number, np.generic, np.ndarray)):
        array = np.asarray(value)
    else:
        raise TypeError(
            f"Unsupported type {type(value).__name__} for metadata column "
            f"{name!r}; supported values are strings, booleans, numbers, "
            "numpy scalars and arrays, and astropy Quantities"
        )
    if array.dtype == np.dtype(object) or array.dtype.names is not None:
        raise TypeError(
            f"Metadata column {name!r} has unsupported object or structured "
            f"dtype {array.dtype}"
        )
    return value if isinstance(value, u.Quantity) else array
