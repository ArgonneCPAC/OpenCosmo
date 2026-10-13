from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Self,
)
from warnings import warn

import opencosmo as oc
from opencosmo.collection.healpix_map import operations as mapops
from opencosmo.collection.healpix_map.state import HealpixMapState
from opencosmo.dataset import state as st

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.io import fits

    from opencosmo.column.column import Column, ColumnMask, ConstructedColumn
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.schema import Schema
    from opencosmo.spatial import Region


class HealpixMap:
    """
    A HealpixMap contains a map-format dataset, typically integrated over a
    specified redshift range. Each column is one layer of the map. The
    HealpixMap object provides an API identical to the standard
    Dataset API, however the data that is provided is returned in healpix or healsparse
    format, which are different than other opencosmo datasets. This also contains some
    convenience functions for standard operations.
    """

    def __init__(self, state: HealpixMapState):
        self.__state = state

    @property
    def _state(self) -> HealpixMapState:
        """Return the internal state for collection and I/O implementations."""
        return self.__state

    def __getattr__(self, key: str):
        try:
            return self.header.parameters[key]
        except KeyError:
            return object.__getattribute__(self, key)

    def __dir__(self):
        return list(self.header.parameters.keys()) + super().__dir__()

    @property
    def nside(self):
        """
        The healpix nside resolution parameter for this map

        Returns
        -------
        dtype: int
        """
        return self.__state.nside

    @property
    def pixels(self):
        """
        The healpix pixels that are included in this map
        """
        return self.__state.pixels

    @property
    def nside_lr(self):
        """
        The low resolution nside resolution parameter used to
        access this map with healsparse.
        Returns
        -------
        dtype: int
        """
        return self.__state.nside_lr

    @property
    def ordering(self):
        """
        The order of pixelization for the map. Either
        NESTED or RING. Maps are currently always saved
        in NESTED format.

        Returns
        -------
        dtype: str
        """
        return self.__state.ordering

    @property
    def full_sky(self):
        """
        Whether the map has full-sky coverage or not
        (note if not you must ask for the data in
        healsparse format and not full healpix format)
        Returns
        -------
        dtype: bool
        """
        return self.__state.full_sky

    def __repr__(self):
        """
        A basic string representation of the dataset
        """
        length = len(self)

        if len(self) < 10:
            repr_ds = self
            table_head = ""

        else:
            repr_ds = self.take(10, at="start")
            table_head = "First 10 rows:\n"

        table_repr = repr_ds.data.__repr__()

        # remove the first line
        table_repr = table_repr[table_repr.find("\n") + 1 :]
        head = f"OpenCosmo Healpix Map Dataset (length={length})\n"
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        return head + cosmo_repr + table_head + table_repr

    def __len__(self):
        return len(self.__state)

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        return st.exit_state(self.__state.dataset, *exc_details)

    @property
    def header(self) -> OpenCosmoHeader:
        """
        The header associated with this dataset.

        OpenCosmo headers generally contain information about the original data this
        dataset was produced from, as well as any analysis that was done along
        the way.

        Returns
        -------
        header: opencosmo.header.OpenCosmoHeader

        """
        return self.__state.header

    @property
    def columns(self) -> list[str]:
        """

        The names of the columns in this map.

        Returns
        -------
        columns: list[str]
        """
        return self.__state.columns

    @property
    def descriptions(self) -> dict[str, str | None]:
        """
        Return the descriptions (if any) of the columns in this map as a dictonary.
        Columns without a description will be included in the dictionary with a value
        of None

        Returns
        -------

        descriptions : dict[str, str | None]
            The column descriptions
        """
        return self.__state.descriptions

    @property
    def region(self) -> Region:
        """
        The region this dataset is contained in. If no spatial
        queries have been performed, this will be the full sky for
        lightcone maps.

        Returns
        -------
        region: opencosmo.spatial.Region

        """
        return self.__state.region

    @property
    def z_range(self):
        """
        The redshift range of the data which created this map.

        Returns
        -------
        z_range: tuple[float, float]
        """
        return self.__state.z_range

    def get_data(
        self,
        format: Literal["healsparse", "healpix", "raw"] = "healsparse",
        nside_out: int | None = None,
        **kwargs,
    ):
        """
        Get the data in this dataset as healsparse map or as healpix maps
        (nest-ordered numpy array). Note that a dataset does not load data from
        disk into memory until this function is called. As a result, you should
        not call this function until you have performed any transformations you
        plan to on the data.

        You can get the data in two formats, "healsparse" (the default) and "healpix".
        "healsparse" format will return the data as a healsparse sparse map.
        "healpix" will return the data as a dictionary of numpy arrays. If the map does not
        cover the full sky, these will be masked numpy arrays in which the pixels
        outside the map's coverage are masked and hold ``healpy.UNSEEN``. For map data,
        due to format requirements, no units will be attached to the data itself,
        although these will match the units from the data attributes.


        Parameters
        ----------
        output: str, default="healsparse"
            The format to output the data in

        Returns
        -------
        data: HealsparseMap | Column | dict[str, ndarray] | ndarray
            The data in this dataset.
        """
        if "output" in kwargs:
            warn(
                "The `output` argument of the `get_data` function has been renamed to `format`. Passing the `output` argument will cause a failure in a future version"
            )
            format = kwargs["output"]
        return mapops.get_data(self.__state, format, nside_out)

    @property
    def data(self):
        """
        Return the data in the dataset in healsparse format. The value of this
        attribute is equivalent to the return value of
        :code:`Dataset.get_data("healsparse")`.


        Returns
        -------
        data : HealsparseMap
            The data in the dataset.

        """
        return self.get_data("healsparse")

    def with_resolution(self, nside) -> HealpixMap:
        """
        Return a copy of the map with a new nside resolution.

        The new resolution must be strictly less than the current
        resolution.

        """
        return HealpixMap(mapops.with_resolution(self.__state, nside))

    def make_schema(self, path: str) -> Schema:
        return mapops.make_schema(self.__state, path)

    def bound(self, region: Region, inclusive: bool = False):
        """
        Restrict this map to some subregion. Be default this will
        include all pixels whose centers fall within the subregion. You can additionally
        include pixels that overalp without there centers being within the
        specified region by passing :code:`inclusive=True`

        If trying to query in a circular region, consider using
        :py:meth:`cone_search <opencosmo.HealpixMap.cone_search>` for simplicity.

        Parameters
        ----------
        region: opencosmo.spatial.Region
            The region to query.

        incluive: bool, default = Flase
            Whether to include pixels that overlap but whose centers are not in the region1

        Returns
        -------
        new_map: opencosmo.HealpixMap
            The map including the pixels within the region.

        Raises
        ------
        ValueError
            If the query region does not overlap with the coverage of this map
            in
        """
        return HealpixMap(mapops.bound(self.__state, region, inclusive))

    def cutouts(
        self,
        centers: SkyCoord,
        angular_size: float | u.Quantity,
        npix: int = 64,
    ) -> Generator[fits.HDUList, None, None]:
        """Create square map cutouts centered at sky coordinates.

        Parameters
        ----------
        centers : astropy.coordinates.SkyCoord
            Centers of the requested cutouts.
        angular_size : float or astropy.units.Quantity
            Angular width of each cutout. Values without units are assumed to
            be in degrees. Must be positive.
        npix : int, default=64
            Number of pixels along each side of each output image. Must be
            positive.

        Yields
        ------
        cutout : astropy.io.fits.HDUList
            An in-memory FITS file containing one image extension per map
            column. Image extensions use a celestial TAN WCS.

        Raises
        ------
        ValueError
            If ``angular_size`` has non-angular units, or if ``angular_size``
            or ``npix`` is not positive.
        TypeError
            If ``angular_size`` is not scalar or ``npix`` is not an integer.

        Notes
        -----
        Cutouts are square and use a 64-by-64 pixel grid by default. The output
        uses an equatorial ``RA---TAN``/``DEC--TAN`` WCS and assumes this map
        uses nested HEALPix ordering in the same celestial frame. Output samples
        whose four-pixel interpolation stencil includes an unavailable map
        pixel are set to NaN. A cutout with no map coverage is therefore an
        all-NaN image with valid WCS metadata.

        This method is intended for small postage-stamp cutouts. Very large
        cutouts may be significantly distorted by the TAN projection.
        """
        yield from mapops.cutouts(self.__state, centers, angular_size, npix)

    def cone_search(self, center: tuple | SkyCoord, radius: float | u.Quantity):
        """
        Perform a search for pixels within some angular distance of some
        given point on the sky. This is a convinience function around
        :py:meth:`bound <opencosmo.HealpixMap.bound>` and is exactly
        equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

        This search is inclusive, meaning this will include all pixels which overlap
        with the region by any amount.

        Parameters
        ----------
        center: tuple | SkyCoord
            The center of the region to search. If a tuple and no units are provided
            assumed to be RA and Dec in degrees.

        radius: float | astropy.units.Quantity
            The angular radius of the region to query. If no units are provided,
            assumed to be degrees.

        Returns
        -------
        new_map: opencosmo.HealpixMap
            The pixels in these maps that fall within the given region.

        """
        region = oc.make_cone(center, radius)
        return self.bound(region)

    def box_search(self, p1: tuple | SkyCoord, p2: tuple | SkyCoord):
        """
        Perform a search for objects within some range in RA and Dec.
        This is a convinience function around
        :py:meth:`bound <opencosmo.HealpixMap.bound>` and is exactly
        equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

        This search is inclusive, meaning this will include all pixels which overlap
        with the region by any amount.

        Parameters
        ----------
        p1: tuple | SkyCoord
            One corner of the box. If a tuple and no units are provided
            assumed to be RA and Dec in degrees.

        p2: tuple | SkyCoord
            The opposite corner of the box. If a tuple and no units are provided,
            assumed to be degrees.

        Returns
        -------
        new_map: opencosmo.HealpixMap
            The pixels in these maps that fall within the given region.

        """
        region = oc.make_skybox(p1, p2)
        return self.bound(region)

    def evaluate(
        self,
        func: Callable,
        format: str = "numpy",
        vectorize=False,
        insert=True,
        **evaluate_kwargs,
    ):
        """
        Iterate over the rows in this collection, apply `func` to each, and collect
        the result as new columns in the dataset. You may also choose to simply return thevalues
        instead of inserting them as a column

        This function is the equivalent of :py:meth:`with_new_columns <opencosmo.HealpixMap.with_new_columns>`
        for cases where the new column is not a simple algebraic combination of existing columns. Unlike
        :code:`with_new_columns`, this method will evaluate the results immediately and the resulting
        columns will not change under unit transformations.

        The function should take in arguments with the same name as the columns in this dataset that
        are needed for the computation, and should return a dictionary of output values.
        The dataset will automatically select the needed columns to avoid unnecessarily reading
        data from disk. The new columns will have the same names as the keys of the output dictionary
        See :ref:`Evaluating On Datasets` for more details.

        If vectorize is set to True, the full columns will be pased to the dataset. Otherwise,
        rows will be passed to the function one at a time.

        This function behaves identically to :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`

        Parameters
        ----------
        func: Callable
            The function to evaluate on the rows in the dataset.

        format: str, default = "numpy"
            The format of the data that is provided to your function. If "astropy", will be a dictionary of
            astropy quantities. If "numpy", will be a dictionary of numpy arrays.

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False)

        insert: bool, default = True
            If true, the data will be inserted as a column in this dataset. Otherwise the data will be returned.

        Returns
        -------
        dataset : HealpixMap
            The new lightcone dataset with the evaluated column(s)
        """
        result = mapops.evaluate(
            self.__state, func, format, vectorize, insert, **evaluate_kwargs
        )
        if isinstance(result, HealpixMapState):
            return HealpixMap(result)
        return result

    def filter(self, *masks: ColumnMask, **kwargs) -> HealpixMap:
        """
        Filter the map based on some criteria. See :ref:`Querying Based on Column
        Values` for more information.

        Parameters
        ----------
        *masks : Mask
            The masks to apply to dataset, constructed with :func:`opencosmo.col`

        Returns
        -------
        dataset : Dataset
            The new dataset with the masks applied.

        Raises
        ------
        ValueError
            If the given  refers to columns that are
            not in the dataset, or the  would return zero rows.

        """
        return HealpixMap(mapops.filter(self.__state, *masks))

    def rows(self) -> Generator[dict[str, float | u.Quantity], None, None]:
        """
        Iterate over the pixels in the map, returning their individual values.
        Rows are returned as a dictionary. For performance, it is recommended
        to first select the columns you need to work with.

        Yields
        -------
        row : dict
            A dictionary of values for each row in the dataset with units.
        """
        yield from mapops.rows(self.__state)

    def select(
        self, *columns: str | Iterable[str], **derived_columns: ConstructedColumn
    ) -> HealpixMap:
        """
        Create a new map from a subset of columns in this map. This function
        behaves the same as :py:meth:`Dataset.select <opencosmo.Dataset.select>`.

        Parameters
        ----------
        *columns : str or list[str]
            The column or columns to select.
        **derived_columns: Column
            Any new derived columns that will be instantiated as part of the select

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected columns.

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        return HealpixMap(mapops.select(self.__state, *columns, **derived_columns))

    def drop(self, columns: str | Iterable[str]) -> HealpixMap:
        """
        Produce a new dataset by dropping columns from this map.

        Parameters
        ----------
        columns : str or list[str]
            The column or columns to drop.

        Returns
        -------
        dataset : Dataset
            The new dataset without the dropped columns

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        return HealpixMap(mapops.drop(self.__state, columns))

    def take(self, n: int, at: str = "random") -> HealpixMap:
        """
        Create a new dataset from some number of rows from this map.

        Can take the first n rows, the last n rows, or n random rows
        depending on the value of 'at'.

        Parameters
        ----------
        n : int
            The number of rows to take.
        at : str
            Where to take the rows from. One of "start", "end", or "random".
            The default is "random".

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected rows.

        Raises
        ------
        ValueError
            If n is negative or greater than the number of rows in the dataset,
            or if 'at' is invalid.

        """
        return HealpixMap(mapops.take(self.__state, n, at))

    def take_range(self, start: int, end: int):
        return HealpixMap(mapops.take_range(self.__state, start, end))

    def take_rows(self, rows: np.ndarray):
        """
        Take the rows of a map specified by the :code:`rows` argument.
        :code:`rows` should be an array of integers. Note that for healpix
        maps the rows refers to the pixel indices.

        Parameters
        ----------
        rows : np.ndarray[int]
            The indices of the rows to take.

        Returns
        -------
        dataset: The dataset with only the specified rows included

        Raises:
        -------
        ValueError:
            If any of the indices is less than 0 or greater than the length of the
            map.

        """
        return HealpixMap(mapops.take_rows(self.__state, rows))

    def with_new_columns(
        self,
        descriptions: str | dict[str, str] | None = None,
        **columns: Column | np.ndarray | u.Quantity,
    ):
        """
        Create a new map with additional columns. These new columns can be derived
        from columns already in the dataset, or a numpy array.  See :ref:`Adding Custom Columns`
        and :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>`
        for examples.

        Parameters
        ----------
        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`HealpixMap.descriptions <opencosmo.HealpixMap.descriptions>`. If a dictionary,
            should have keys matching the column names.

        ** columns : opencosmo.Column | np.ndarray | u.quantity
            The new columns

        Returns
        -------
        dataset : opencosmo.Dataset
            This dataset with the columns added

        """
        if descriptions is None:
            descriptions = {}
        return HealpixMap(
            mapops.with_new_columns(self.__state, descriptions, **columns)
        )

    def sort_by(self, column: str, invert: bool = False):
        """
        Sort this map by the values in a given column. By default sorting is in
        ascending order (least to greatest). Pass invert = True to sort in descending
        order (greatest to least).

        This is not generally particular useful in map queries, but can be used to
        enforce ordering schemes or find outlier pixels.

        Parameters
        ----------
        column : str
            The column in the map dataset to
            order the collection by.

        invert : bool, default = False
            If False (the default), ordering will be from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : Dataset
            A new Dataset ordered by the given column.


        """
        return HealpixMap(mapops.sort_by(self.__state, column, invert))

    def with_units(
        self,
        convention: str | None = None,
        conversions: dict[u.Unit, u.Unit] | None = None,
        **columns: u.Unit,
    ) -> Self:
        r"""
        Unit conversion is usually supported for OpenCosmo datasets, however maps tend to be integrated
        quantities over a range of redshifts which correspond to observed units so applying unit conversions
        is not generally easy or appropriate.
        """

        raise NotImplementedError(
            "Unit conversions not supported on maps, these are integrated over redshift so conversions are non-trivial!"
        )
