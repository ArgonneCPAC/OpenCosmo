from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)
from warnings import warn

from astropy.coordinates import SkyCoord

import opencosmo as oc
from opencosmo import facade
from opencosmo.collection.healpix_map import operations as mapops
from opencosmo.collection.healpix_map.healpix_map import HealpixMap
from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone import state as lcst
from opencosmo.dataset import state as st
from opencosmo.deprecated import deprecated

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Iterator,
        KeysView,
    )
    from uuid import UUID

    import astropy.units as u  # type: ignore
    import numpy as np
    import numpy.typing as npt
    from astropy.io import fits

    from opencosmo.column.column import (
        ColumnMask,
        ConstructedColumn,
    )
    from opencosmo.dataset import Dataset
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema
    from opencosmo.spatial import Region


class Lightcone:
    """
    A lightcone contains two or more datasets that are part of a lightcone. Typically
    each dataset will cover a specific redshift range. The Lightcone object
    hides these details, providing an API that is identical to the standard
    Dataset API. Additionally, the lightcone contains some convinience functions
    for standard operations.

    Lightcones can be nested. In this case, the top level will split the datasets
    up by step, while the second level will split the datasets up by type. This nested
    scheme (at present) is used for Diffsky catalogs, which may contain both cores
    and synthetic cores that need to be adddressed (and more importantly, written)
    seperately from one another.
    """

    def __init__(self, state: lcst.LightconeState):
        self.__state = state

    def __getitem__(self, key: int | str) -> Dataset | Lightcone:
        return facade.wrap(lcst.view(self.__state, key))

    def __iter__(self) -> Iterator[int | str]:
        return iter(lcst.public_keys(self.__state))

    def __contains__(self, key: object) -> bool:
        return key in lcst.public_keys(self.__state)

    def keys(self) -> KeysView[int | str]:
        return dict.fromkeys(lcst.public_keys(self.__state)).keys()

    def values(self) -> list[Dataset | Lightcone]:
        return [self[key] for key in self]

    def items(self) -> list[tuple[int | str, Dataset | Lightcone]]:
        return [(key, self[key]) for key in self]

    @property
    def _state(self) -> lcst.LightconeState:
        """Return the internal state for collection and I/O implementations."""
        return self.__state

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

        table_repr = repr_ds.get_data().__repr__()
        # remove the first line
        table_repr = table_repr[table_repr.find("\n") + 1 :]
        z_range = self.z_range
        head = (
            f"OpenCosmo Lightcone Dataset (length={length}, "
            f"{z_range[0]} < z < {z_range[1]})\n"
        )
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        return head + cosmo_repr + table_head + table_repr

    def __len__(self):
        return len(self.__state)

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for _, leaf in self.__state.leaves:
            st.exit_state(leaf, *exc_details)

    def __getattr__(self, key: str):
        try:
            return self.__state.header.parameters[key]
        except KeyError:
            return object.__getattribute__(self, key)

    def __dir__(self):
        return list(self.header.parameters.keys()) + super().__dir__()

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
    def scope(self):
        """The lightcone-level scope for scoped columns."""
        return self.__state.scope

    @property
    def columns(self) -> list[str]:
        """
        The names of the columns in this dataset.

        Returns
        -------
        columns: list[str]
        """
        return self.__state.columns

    @property
    def map_columns(self) -> list[str] | None:
        """
        If this lightcone has a map, return the columns
        in the map. If the lightcone does not have a map, returns None.

        Returns
        -------
        map_columns: list[str] | None
        """
        if self.__state.maps is None:
            return None
        return self.__state.maps.columns

    @property
    def map(self) -> HealpixMap | None:
        if self.__state.maps is None:
            return None
        return HealpixMap(self.__state.maps)

    # Internal identity used by link/mapping resolution.
    @property
    def uuid(self) -> UUID:
        return self.__state.uuid

    @property
    def descriptions(self) -> dict[str, str | None]:
        """
        Return the descriptions (if any) of the columns in this lightcone as a dictonary.
        Columns without a description will be included in the dictionary with a value
        of None

        Returns
        -------

        descriptions : dict[str, str | None]
            The column descriptions
        """
        return self.__state.descriptions

    @property
    def units(self) -> dict[str, u.Unit | None]:
        """
        Return the units of the columns in this lightcone. Columns without a unit will
        return a value of None

        Returns
        -------

        descriptions : dict[str, str | None]
            The column descriptions
        """
        return self.__state.units

    @property
    def region(self) -> Region | None:
        """
        The region this dataset is contained in. If no spatial
        queries have been performed, this will be the entire
        simulation box for snapshots or the full sky for lightcones

        Returns
        -------
        region: opencosmo.spatial.Region

        """
        return self.__state.region

    @property
    def sorted_by(self) -> str | None:
        """
        The column this dataset is sorted by. If not sorted, returns None.

        Returns
        -------
        column: Optional[str]
        """
        return self.__state.sort_key[0] if self.__state.sort_key is not None else None

    def get_pixels(self, nside: int = 64):
        """
        Return the HEALPix pixels occupied by this lightcone at a given resolution.

        Pixel indices are returned in nested ordering. The ``nside`` parameter
        controls angular resolution: larger values produce finer pixels. The
        requested resolution may not exceed the resolution of the spatial index
        stored in the file.

        Parameters
        ----------
        nside : int, default = 64
            The HEALPix resolution parameter. Must be a positive power of two.

        Returns
        -------
        pixels : numpy.ndarray[int]
            HEALPix pixel indices (nested ordering) occupied by this lightcone
            at the given resolution.

        Raises
        ------
        ValueError
            If ``nside`` is not a positive power of two, if ``nside`` exceeds
            the maximum resolution of the spatial index, or if the lightcone
            does not have a spatial index.
        """
        return lcops.get_pixels(self.__state, nside)

    def get_data(
        self,
        format="astropy",
        unpack: bool = True,
        wrap_single: bool = False,
        **kwargs,
    ):
        """
        Get the data in this dataset as an astropy table/column or as
        numpy array(s). Note that a dataset does not load data from disk into
        memory until this function is called. As a result, you should not call
        this function until you have performed any transformations you plan to
        on the data.

        You can get the data in two formats, "astropy" (the default) and "numpy".
        "astropy" format will return the data as an astropy table with associated
        units. "numpy" will return the data as a dictionary of numpy arrays. The
        numpy values will be in the associated unit convention, but no actual
        units will be attached.

        If the dataset only contains a single column, it will be returned as an
        astropy.table.Column or a single numpy array. Pass :code:`wrap_single=True`
        to always return the format's multi-column container (QTable, DataFrame,
        dict, ...) regardless of column count.

        Parameters
        ----------
        output: str, default="astropy"
            The format to output the data in. Currently supported are "astropy", "numpy",
            "pandas", "polars", and "arrow"

        wrap_single: bool, default=False
            If True, always return the format's natural multi-column container even
            when only one column is present.

        Returns
        -------
        data: Table | Column | dict[str, ndarray] | ndarray
            The data in this dataset.
        """
        if "output" in kwargs:
            warn(
                "The `output` argument of the `get_data` function has been renamed to `format`. Passing the `output` argument will cause a failure in a future version"
            )
            format = kwargs["output"]
        return lcops.get_data(
            self.__state,
            format,
            unpack,
            wrap_single,
            ignore_sort=kwargs.get("ignore_sort", False),
        )

    @property
    @deprecated(
        msg="Accessing data through the .data attribute is deprecated and will be removed in a future version. Use get_data()",
    )
    def data(self):
        """
        Return the data in the dataset in astropy format. The value of this
        attribute is equivalent to the return value of
        :code:`Dataset.get_data("astropy")`.

        Returns
        -------
        data : astropy.table.Table or astropy.table.Column
            The data in the dataset.

        """
        return self.get_data("astropy")

    def cutouts(
        self, size: float | u.Quantity, npix: int = 64
    ) -> Generator[tuple[dict[str, float | u.Quantity], fits.HDUList], None, None]:
        """Create square map cutouts centered on the catalog objects.

        The map opened alongside this lightcone is sampled around each object's
        ``ra`` and ``dec`` coordinates. Each result contains the catalog row and
        a FITS HDU list holding the corresponding square cutout.

        Parameters
        ----------
        size : float or astropy.units.Quantity
            Angular width of each cutout. Values without units are assumed to
            be in degrees. Must be positive.
        npix : int, default=64
            Number of pixels along each side of the output image. Must be
            positive.

        Yields
        ------
        row : dict
            The catalog row at the center of the cutout.
        cutout : astropy.io.fits.HDUList
            An in-memory FITS file containing one image extension per map
            column. Image extensions use a celestial TAN WCS.

        Raises
        ------
        ValueError
            If no map was opened with the lightcone, if ``size`` has
            non-angular units, or if ``size`` or ``npix`` is not positive.
        TypeError
            If ``size`` is not a scalar number or quantity, or ``npix`` is not
            an integer.

        Notes
        -----
        Cutouts are square and use a 64-by-64 pixel grid by default. The output
        uses an equatorial ``RA---TAN``/``DEC--TAN`` WCS and assumes the input
        map uses nested HEALPix ordering in the same celestial frame. Output
        samples whose interpolation stencil includes unavailable map pixels are
        set to NaN.
        """
        if self.__state.maps is None:
            raise ValueError("No map was opened with this lightcone")
        missing_coordinates = {"ra", "dec"}.difference(self.columns)
        if missing_coordinates:
            missing = ", ".join(sorted(missing_coordinates))
            raise ValueError(
                f"Lightcone cutouts require ra and dec columns; missing: {missing}"
            )

        coordinates = lcops.get_data(lcops.select(self.__state, "ra", "dec"))
        centers = SkyCoord(**coordinates)
        cutouts = mapops.cutouts(self.__state.maps, centers, size, npix=npix)

        for entry, cutout in zip(lcops.rows(self.__state), cutouts):
            yield entry, cutout

    def with_redshift_range(self, z_low: float, z_high: float):
        """
        Restrict this lightcone to a specific redshift range. Lightcone datasets will
        always contain a column titled "redshift." This function is always operates on
        this column.

        This function also updates the value in
        :py:meth:`Lightcone.z_range <opencosmo.collection.Lightcone.z_range>`,
        so you should always use it rather than filteringo n the column directly.
        """
        return Lightcone(lcops.with_redshift_range(self.__state, z_low, z_high))

    def make_schema(
        self, path: str, _min_size=100_000, no_stack: bool = False
    ) -> Schema:
        return lcops.make_schema(self.__state, path, _min_size, no_stack)

    def bound(self, region: Region, select_by: str | None = None):
        """
        Restrict the dataset to some subregion. The subregion will always be evaluated
        in the same units as the current dataset. For example, if the dataset is
        in the default "comoving" unit convention, positions are always in units of
        comoving Mpc. However Region objects themselves do not carry units.
        See :doc:`spatial_ref` for details of how to construct regions.

        Parameters
        ----------
        region: opencosmo.spatial.Region
            The region to query.

        Returns
        -------
        dataset: opencosmo.Dataset
            The portion of the dataset inside the selected region

        Raises
        ------
        ValueError
            If the query region does not overlap with the region this dataset resides
            in
        AttributeError:
            If the dataset does not contain a spatial index
        """
        return Lightcone(lcops.bound(self.__state, region, select_by))

    def cone_search(self, center: tuple | SkyCoord, radius: float | u.Quantity):
        """
        Perform a search for objects within some angular distance of some
        given point on the sky. This is a convinience function around
        :py:meth:`bound <opencosmo.Lightcone.bound>` and is exactly
        equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

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
        new_lightcone: opencosmo.Lightcone
            The rows in this lightcone that fall within the given region.

        """
        region = oc.make_cone(center, radius)
        return self.bound(region)

    def box_search(self, p1: tuple | SkyCoord, p2: tuple | SkyCoord):
        """
        Perform a box search in a given RA and Dec range. Of course this is not
        really a "box" on the surface of a sphere, but it's the closet thing we got.
        This metbhod is exactly equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

        Parameters
        ----------

        p1: tuple | astropy.coordinates.SkyCoord
            A point defining one corner of the box. If a tuple with no units, values are assumed to
            be RA and Dec in degrees.
        p2: tuple | astropy.coordinates.SkyCoord
            A point defining the opposite corner of the box. If a tuple with no units, values are assumed to
            be RA and Dec in degrees.

        Returns
        -------

        new_lightcone: opencosmo.Lightcone
            The new dataset, only including data within the specified region.
        """
        region = oc.make_skybox(p1, p2)
        return self.bound(region)

    def pixel_search(self, pixels: npt.NDArray[np.int_], nside: int = 64):
        """
        Return the subset of this lightcone that falls within a set of HEALPix pixels.

        Pixels must be specified in nested ordering and must be valid indices at
        the given ``nside``. Duplicate pixel indices are ignored. Use
        :py:meth:`get_pixels <opencosmo.Lightcone.get_pixels>` to discover
        which pixels this lightcone covers.

        Parameters
        ----------
        pixels : array_like[int]
            HEALPix pixel indices to query, in nested ordering. Must be a 1-D
            array of non-negative integers. Values must be less than
            ``healpy.nside2npix(nside)``.
        nside : int, default = 64
            The HEALPix resolution parameter. Must be a positive power of two
            and must not exceed the resolution of the spatial index stored in
            the file.

        Returns
        -------
        lightcone : opencosmo.Lightcone
            A new lightcone containing only the objects that fall within the
            specified pixels.

        Raises
        ------
        ValueError
            If ``nside`` is not a positive power of two, or if ``pixels``
            contains values that are out of range for the given ``nside``.
        """
        return Lightcone(lcops.pixel_search(self.__state, pixels, nside))

    def evaluate(
        self,
        func: Callable,
        vectorize=False,
        insert=True,
        format: str = "astropy",
        batch_size: int = -1,
        allow_overwrite: bool = False,
        combine_mode: Literal["concatenate", "sum", "prod", "avg"] = "concatenate",
        **evaluate_kwargs,
    ):
        """
        Iterate over the rows in this collection, apply `func` to each, and collect
        the result as new columns in the dataset. You may also choose to simply return thevalues
        instead of inserting them as a column

        This function is the equivalent of :py:meth:`with_new_columns <opencosmo.Lightcone.with_new_columns>`
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

        If a :code:`batch_size` is set, opencosmo will pass data to your function in batches of rows. In a lightcone,
        batches may be smaller than the given chunk size but will never be larger. Exact batch sizes
        will depend on the layout of the lightcone. Setting a batch size overrides the :code:`vectorize`
        flag.

        This function behaves (mostly) identically to :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`

        Parameters
        ----------
        func: Callable
            The function to evaluate on the rows in the dataset.

        format: str, default = "astropy"
            The format in which to provide column data to your function. Supports the same formats
            as :py:meth:`get_data <opencosmo.Lightcone.get_data>` ("astropy", "numpy", "pandas",
            "polars", "arrow", "jax"). When :code:`insert=True`, the function's output is converted
            back to numpy before being stored.

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False)

        insert: bool, default = True
            If true, the data will be inserted as a column in this dataset. Otherwise the data will be returned.

        combine_mode: str, "concatenate", "sum", "prod", or "avg", default = "concatenate"
            How to combine results from the lightcone's underlying datasets when
            ``insert=False``. ``avg`` weights each underlying result by the number
            of rows in its dataset. All underlying results must have the same length
            for modes other than ``concatenate``. This argument has no effect when
            ``insert=True``.

        Returns
        -------
        dataset : Lightcone
            The new lightcone dataset with the evaluated column(s)
        """
        result = lcops.evaluate(
            self.__state,
            func,
            vectorize,
            insert,
            format,
            batch_size,
            allow_overwrite,
            combine_mode,
            **evaluate_kwargs,
        )
        if isinstance(result, lcst.LightconeState):
            return Lightcone(result)
        return result

    def filter(self, *masks: ColumnMask, mode: str = "global", **kwargs) -> Lightcone:
        """
        Filter the dataset based on some criteria. See :ref:`Querying Based on Column
        Values` for more information.

        Parameters
        ----------
        *masks : Mask
            The masks to apply to dataset, constructed with :func:`opencosmo.col`

        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions (e.g. ``oc.col("x").mean()``) used
            inside the masks are computed when running under MPI. Defaults to
            ``"global"`` so every rank filters against the same cross-rank
            scalar; pass ``"local"`` to use each rank's per-rank scalar.

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
        return Lightcone(lcops.filter(self.__state, *masks, mode=mode, **kwargs))

    def rows(self) -> Generator[dict[str, float | u.Quantity], None, None]:
        """
        Iterate over the rows in the dataset. Rows are returned as a dictionary
        For performance, it is recommended to first select the columns you need to
        work with.

        Yields
        -------
        row : dict
            A dictionary of values for each row in the dataset with units.
        """
        yield from lcops.rows(self.__state)

    def select(
        self,
        *columns: str | Iterable[str],
        mode: str = "global",
        **derived_columns: ConstructedColumn,
    ) -> Lightcone:
        """

        Create a new lightcone dataset from a subset of columns in this lightcone dataset.
        This function accepts wildcards. For exampe, "lsst*" will select all columns
        that start with "lsst", while "*host*" will select all columns that have
        "host" somewhere in the middle.

        You can also create new columns as part of this call, as long as they are
        derived from other columns in the dataset. For example:

        .. code-block:: python

            import opencosmo as oc
            from opencosmo.column import add_mag_cols

            dataset = oc.open("galaxy_catalog.hdf5")
            total_mag = add_mag_cols("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
            # Note, you can also use oc.col to do this manually


            dataset = dataset.select("ra", "dec", "*host*", "lsst*", lsst_total = total_mag)

        This new dataset will contain the :code:`ra` and :code:`dec` columns, all the columns
        with :code:`host` somewhere in the name, all the columns that start with :code:`lsst`
        and a newly-constructed :code:`lsst_total` column.



        Parameters
        ----------
        *columns : str or list[str]
            The column or columns to select.

        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions inside derived column expressions
            are computed when running under MPI. Defaults to ``"global"``
            (cross-rank); pass ``"local"`` for per-rank scalars.

        **derived_columns : Column
            Additional columns to create as part of the selection.

        Returns
        -------
        dataset : Lightcone
            The new lightcone with only the selected columns.

        Raises
        ------
        ValueError
            If any of the required columns are not in the dataset.
        """
        return Lightcone(
            lcops.select(self.__state, *columns, mode=mode, **derived_columns)
        )

    def drop(self, *columns: str | Iterable[str]) -> Lightcone:
        """
        Produce a new dataset by dropping columns from this dataset.

        Parameters
        ----------
        *columns : str or list[str]
            The column or columns to drop.

        Returns
        -------
        dataset : Lightcone
            The new lightcone without the dropped columns

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        return Lightcone(lcops.drop(self.__state, *columns))

    def take(
        self, n: int, at: str = "random", mode: Literal["local", "global"] = "local"
    ) -> Lightcone:
        """
        Create a new dataset from some number of rows from this dataset.

        Can take the first n rows, the last n rows, or n random rows
        depending on the value of 'at'.

        Parameters
        ----------
        n : int
            The number of rows to take.
        at : str
            Where to take the rows from. One of "start", "end", or "random".
            The default is "random".
        mode : str, "local" or "global", default = "local"
            Controls how ``n`` is interpreted when running under MPI. Has no
            effect if you are not using MPI.

            * ``"local"`` (default): ``n`` rows are taken independently on
              each rank.
            * ``"global"``: ``n`` is the total number of rows to select across
              all ranks combined. Each rank receives the portion of those rows
              that it owns. If the dataset is sorted, ranks will coordinate
              to take from the globally-sorted dataset.

        Returns
        -------
        lightcone : Lightcone
            The new lightcone with only the selected rows.

        Raises
        ------
        ValueError
            If n is negative or greater than the number of rows in the dataset,
            or if 'at' is invalid.

        """
        return Lightcone(lcops.take(self.__state, n, at, mode))

    def take_range(
        self, start: int, end: int, mode: Literal["local", "global"] = "local"
    ):
        """
        Create a new lightcone from a row range in this lightcone. We use standard
        indexing conventions, so the rows included will be start -> end - 1. Because
        lightcones are stacked by redshift, this operation effectively takes a
        redshift range. If you know the exact redshift range you want, use
        :py:meth:`with_redshift_range <opencosmo.Lightcone.with_redshift_range>`.

        Parameters
        ----------
        start : int
            The beginning of the range.
        end : int
            The end of the range (exclusive).
        mode : str, "local" or "global", default = "local"
            Controls how ``start`` and ``end`` are interpreted when running
            under MPI. Has no effect if you are not using MPI.

            * ``"local"`` (default): the range is applied independently on
              each rank.
            * ``"global"``: ``start`` and ``end`` index into the global row
              space across all ranks combined. Each rank receives the portion
              of that range it owns. If the lightcone is sorted, ranks will
              coordinate to take from the globally-sorted lightcone.

        Returns
        -------
        lightcone : opencosmo.Lightcone
            The lightcone with only the specified range of rows.

        Raises
        ------
        ValueError
            If start or end are negative or greater than the length of the dataset
            or if end is greater than start.

        """
        return Lightcone(lcops.take_range(self.__state, start, end, mode))

    def take_rows(self, rows: DataIndex):
        """
        Take the rows of a lightcone specified by the :code:`rows` argument.
        :code:`rows` should be an array of integers.

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
            lightcone.

        """
        return Lightcone(lcops.take_rows(self.__state, rows))

    def with_new_columns(
        self,
        descriptions: str | dict[str, str] = {},
        allow_overwrite: bool = False,
        mode: str = "global",
        **columns: ConstructedColumn | np.ndarray | u.Quantity,
    ):
        """
        Create a new dataset with additional columns. These new columns can be derived
        from columns already in the dataset, a numpy array, or an Astropy quantity
        array. When a column is derived from other columns, it will behave
        appropriately under unit transformations. See :ref:`Adding Custom Columns`
        and :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>`
        for examples.

        Parameters
        ----------
        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`Lightcone.descriptions <opencosmo.Lighcone.descriptions>`. If a dictionary,
            should have keys matching the column names.

        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions nested inside derived column
            expressions are computed when running under MPI. Defaults to
            ``"global"`` (cross-rank); pass ``"local"`` for per-rank scalars.

        ** columns : opencosmo.Column | np.ndarray | u.quantity
            The new columns

        Returns
        -------
        dataset : opencosmo.Dataset
            This dataset with the columns added

        """
        return Lightcone(
            lcops.with_new_columns(
                self.__state, descriptions, allow_overwrite, mode=mode, **columns
            )
        )

    def sort_by(self, column: str | None, invert: bool = False):
        """
        Sort this dataset by the values in a given column. By default sorting is in
        ascending order (least to greatest). Pass invert = True to sort in descending
        order (greatest to least).

        This can be used to, for example, select largest halos in a given
        dataset:

        .. code-block:: python

            dataset = oc.open("haloproperties.hdf5")
            dataset = dataset
                        .sort_by("fof_halo_mass")
                        .take(100, at="start")

        Parameters
        ----------
        column : Optional[str]
            The column in the halo_properties or galaxy_properties dataset to
            order the collection by. Pass None to remove sorting.

        invert : bool, default = False
            If False (the default), ordering will be from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : Dataset
            A new Dataset ordered by the given column.


        """

        return Lightcone(lcops.sort_by(self.__state, column, invert))

    def with_units(
        self,
        convention: str | None = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns: u.Unit,
    ) -> Lightcone:
        r"""
        Create a new lightcone from this one with a different unit convention or
        with certain columns converted to a different compatible unit.

        Unit conversions are always performed after a change of convention, and
        changing conventions clears any existing unit conversions.

        For more, see :doc:`units`.

        .. code-block:: python

            import astropy.units as u

            # this works
            lc = lc.with_units(fof_halo_mass=u.kg)

            # this clears the previous conversion
            lc = lc.with_units("scalefree")

            # This now fails, because the units of masses
            # are Msun / h, which cannot be converted to kg
            lc = lc.with_units(fof_halo_mass=u.kg)

            # this will now work, wince the units of halo mass in the "physical"
            # convention are Msun (no h).
            lc = lc.with_units("physical", fof_halo_mass=u.kg, fof_halo_center_x=u.lyr)

            # Suppose you want your distances in lightyears, but the x coordinate of your
            # halo center in kilometers, for some reason ¯\_(ツ)_/¯
            blanket_conversions = {u.Mpc: u.lyr}
            lc = lc.with_units(conversions = blanket_conversions, fof_halo_center_x = u.km)

        Parameters
        ----------
        convention : str, optional
            The unit convention to use. One of "physical", "comoving",
            "scalefree", or "unitless".

        conversions: dict[astropy.units.Unit, astropy.units.Unit]
            Conversions that apply to all columns in the lightcone with the
            unit given by the key.

        **column_conversions: astropy.units.Unit
            Custom unit conversions for specific columns
            in this dataset.

        Returns
        -------
        lightcone : Lightcone
            The new lightcone with the requested unit convention and/or conversions.
        """
        return Lightcone(
            lcops.with_units(self.__state, convention, conversions, **columns)
        )
