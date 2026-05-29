from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    Optional,
    TypeAlias,
)
from warnings import warn

import astropy.units as u  # type: ignore
import numpy as np
from astropy.table import QTable  # type: ignore
from deprecated.sphinx import deprecated

import opencosmo.dataset.state as st
from opencosmo.column import Column
from opencosmo.dataset.formats import verify_format
from opencosmo.index import get_range

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology

    from opencosmo.column.column import Column, ColumnMask, ConstructedColumn
    from opencosmo.dataset.state import DatasetState
    from opencosmo.dtypes import HaccSimulationParameters
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region


OpenCosmoData: TypeAlias = QTable | u.Quantity | dict[str, np.ndarray] | np.ndarray


class Dataset:
    """
    User-facing wrapper around a :class:`DatasetState`. ``Dataset`` only
    sanitizes user input (validates formats, expands wildcards, normalizes
    descriptions) and forwards to the canonical state functions in
    :mod:`opencosmo.dataset.state`. The state is the single source of truth
    for the header, region, spatial tree, and column data.

    Collections hold ``DatasetState`` instances directly and only construct
    a ``Dataset`` at the public API boundary.
    """

    def __init__(self, state: DatasetState):
        self.__state = state

    @property
    def state(self) -> DatasetState:
        """
        The underlying :class:`DatasetState`. Intended for internal callers
        (collections, I/O glue) that need to drop down to the canonical state
        API. Will be removed once the I/O layer returns states directly.
        """
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
        head = f"OpenCosmo Dataset (length={length})\n"
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        return head + cosmo_repr + table_head + table_repr

    def __len__(self):
        return len(self.__state)

    def __enter__(self):
        # Need to write tests
        return self

    def __exit__(self, *exc_details):
        return st.exit_state(self.__state, *exc_details)

    def close(self):
        return st.exit_state(self.__state)

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
        The names of the columns in this dataset.

        Returns
        -------
        columns: list[str]
        """
        return self.__state.columns

    @property
    def descriptions(self) -> dict[str, Optional[str]]:
        """
        Return the descriptions (if any) of the columns in this dataset as a dictonary.
        Columns without a description will be included in the dictionary with a value
        of None

        Returns
        -------

        descriptions : dict[str, str | None]
            The column descriptions
        """
        return self.__state.descriptions

    @property
    def units(self) -> dict[str, Optional[u.Unit]]:
        """
        Return the current units of all columns in the dataset. Columns without units will
        return None.

        Returns
        -------

        descriptions : dict[str, str | None]
            The column units

        """
        return self.__state.units

    @property
    def cosmology(self) -> Cosmology:
        """
        The cosmology of the simulation this dataset is drawn from as
        an astropy.cosmology.Cosmology object.

        Returns
        -------
        cosmology: astropy.cosmology.Cosmology
        """
        return self.__state.header.cosmology

    @property
    def dtype(self) -> str:
        """
        The data type of this dataset.

        Returns
        -------
        dtype: str
        """
        return str(self.__state.header.file.data_type)

    @property
    def redshift(self) -> float | tuple[float, float] | None:
        """
        The redshift slice or range this dataset was drawn from

        Returns
        -------
        redshift: float

        """
        return self.__state.header.file.redshift

    @property
    def region(self) -> Region:
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
    def simulation(self) -> Optional[HaccSimulationParameters]:
        """
        The parameters of the simulation this dataset is drawn
        from. May return None if the parameters are not included
        in the file

        Returns
        -------
        parameters: Optional[opencosmo.dtypes.hacc.HaccSimulationParameters]
        """
        return getattr(self.__state.header, "simulation", None)

    @property
    def sorted_by(self) -> Optional[str]:
        """
        The column this dataset is sorted by. If not sorted, returns None.

        Returns
        -------
        column: Optional[str]
        """
        return self.__state.sort_key[0] if self.__state.sort_key is not None else None

    @property
    @deprecated(
        version="1.1.0",
        reason="Accessing data through the .data attribute is deprecated and will be removed in a future version. Use get_data()",
    )
    def data(self) -> QTable | u.Quantity:
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

    def get_data(
        self,
        format="astropy",
        unpack=True,
        metadata_columns=[],
        wrap_single=False,
        **kwargs,
    ) -> OpenCosmoData:
        """
        Get the data in this dataset as an astropy table/column or as
        numpy array(s). Note that a dataset does not load data from disk into
        memory until this function is called. As a result, you should not call
        this function until you have performed any transformations you plan to
        on the data.

        The method supports output into several different formats, including
        "astropy", "numpy", "pandas", "polars", "jax", and "arrow". Although astropy
        and numpy are core dependencies of OpenCosmo, the remaining formats
        require you to have the relevant libraries installed in your python
        environment. This method will check that it can import the necessary
        libraries before attempting to read data. Note that outputting as
        "polars" or "arrow" requires copying the data out of its original
        numpy arrays, which will impact performance.

        If the dataset only contains a single column, it will not be put in a table
        or dictionary. "astropy", "numpy" and "arrow" will return a single array
        in this case, while "polars" and "pandas" will return a Series object. Pass
        :code:`wrap_single=True` to always return the format's multi-column container
        (QTable, DataFrame, dict, ...) regardless of column count.

        Parameters
        ----------
        output: str, default="astropy"
            The format to output the data in.
            Currently supported are "astropy", "numpy", "pandas", "polars", "arrow", "jax"

        wrap_single: bool, default=False
            If True, always return the format's natural multi-column container even
            when only one column is present.

        Returns
        -------
        data: Any
            The data in this dataset.
        """
        if "output" in kwargs:
            warn(
                "The `output` argument of the `get_data` function has been renamed to `format`. Passing the `output` argument will cause a failure in a future version"
            )
            format = kwargs["output"]

        verify_format(format)
        return st.get_data(
            self.__state,
            format=format,
            unpack=unpack,
            metadata_columns=metadata_columns,
            wrap_single=wrap_single,
            **kwargs,
        )

    def bound(self, region: Region, select_by: Optional[str] = None):
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
        return Dataset(st.bound(self.__state, region, select_by))

    def evaluate(
        self,
        func: Callable,
        vectorize=False,
        insert=True,
        format="astropy",
        batch_size: int = -1,
        allow_overwrite: bool = False,
        _verify: bool = True,
        **evaluate_kwargs,
    ) -> Dataset | dict[str, np.ndarray]:
        """
        Iterate over the rows in this dataset, apply :code:`func` to each, and collect
        the result as new columns in the dataset.

        This function is the equivalent of :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`
        for cases where the new column is not a simple algebraic combination of existing columns. Unlike
        :code:`with_new_columns`, this method will evaluate the results immediately and the resulting
        columns will not change under unit transformations. You may also choose to simply return the result
        instead of adding it as a column.

        The function should take in arguments with the same name as the columns in this dataset that
        are needed for the computation, and should return a dictionary of output values. Any addition
        arguments needed by the function can be passed as keyword arguments to :code:`evaluate`.

        The dataset will automatically selected the needed columns to avoid reading unnecessarily reading
        data from disk. The new columns will have the same names as the keys of the output dictionary
        See :ref:`Evaluating On Datasets` for more details. The keys of this dictionary must be different
        from the names of the columns that are already in the dataset, unless allow_overwrite is set
        to :code`True`

        If vectorize is set to True, the full columns will be pased to the dataset. Otherwise,
        rows will be passed to the function one at a time. If the function returns None, this method
        will also return None as output.

        Keyword arguments can be used to pass in external values that are not columns in the dataset.
        For example, we can compute each halo's gas fraction bias — how much gas it retains relative to
        the cosmic baryon fraction — by passing the dataset's cosmology object as a keyword argument:

        .. code-block:: python

            def baryon_fraction_bias(sod_halo_mass_gas, sod_halo_mass, cosmology):
                f_gas = sod_halo_mass_gas / sod_halo_mass
                f_cosmic = cosmology.Ob0 / cosmology.Om0
                return {"sod_halo_baryon_bias": f_gas / f_cosmic}

            ds = ds.evaluate(baryon_fraction_bias, cosmology=ds.cosmology, vectorize=True)

        Parameters
        ----------

        func: Callable
            The function to evaluate on the rows in the dataset.

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False). Ignored if :code:`batch_size` is set.

        insert: bool, default = True
            If true, the data will be inserted as a column in this dataset. The new column will have the same name
            as the function. Otherwise the data will be returned directly.

        format: str, default = astropy
            The format in which to provide column data to your function. Supports the same formats
            as :py:meth:`get_data <opencosmo.Dataset.get_data>` ("astropy", "numpy", "pandas",
            "polars", "arrow", "jax"). When :code:`insert=True`, the function's output is converted
            back to numpy before being stored.
        allow_overwrite: bool, default = False

        batch_size: int, default = -1
            If set, feed data to the function in batches of the specified size. Default is -1, which disables batching. If
            set to another value, the :code:`vectorize` flag is ignored.

        **evaluate_kwargs: any,
            Any additional arguments that are required for your function to run. These will be passed directly
            to the function as keyword arguments. If a kwarg is an array of values with the same length as the dataset,
            it will be treated as an additional column.

        Returns
        -------
        result : Dataset | dict[str, np.ndarray | astropy.units.Quantity]
            The new dataset with the evaluated column(s) or the results as numpy arrays or astropy quantities
        """
        verify_format(format)
        result = st.evaluate(
            self.__state,
            func,
            vectorize=vectorize,
            insert=insert,
            format=format,
            batch_size=batch_size,
            allow_overwrite=allow_overwrite,
            **evaluate_kwargs,
        )
        if not insert:
            return result
        return Dataset(result)

    def filter(self, *masks: ColumnMask) -> Dataset:
        """
        Filter the dataset based on some criteria. See :ref:`Querying Based on Column
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
        return Dataset(st.filter(self.__state, *masks))

    def rows(
        self,
        include_units: bool = True,
        metadata_columns=[],
    ) -> Generator[Mapping[str, float | u.Quantity | np.ndarray]]:
        """
        Iterate over the rows in the dataset. Rows are returned as a dictionary
        For performance, it is recommended to first select the columns you need to
        work with.

        Parameters
        ----------
        output: str, default = "astropy"
            Whether to return values as "astropy" quantities or "numpy" scalars


        Yields
        -------
        row : dict
            A dictionary of values for each row in the dataset with units.

        """
        for row in st.iter_rows(self.__state, metadata_columns):
            output_data = row
            if not isinstance(output_data, dict):
                output_data = {self.columns[0]: row}

            if not include_units:
                output_data = {
                    name: val.value if isinstance(val, u.Quantity) else val
                    for name, val in output_data.items()
                }
            yield output_data

    def select(
        self, *columns: str | Iterable[str], **derived_columns: ConstructedColumn
    ) -> Dataset:
        """
        Create a new dataset from a subset of columns in this dataset. This
        function accepts wildcards. For exampe, "fof*" will select all columns
        that start with "fof", while "*com*" will select all columns that have
        "com" somewhere in the middle.

        You can also create new columns as part of this call, as long as they are
        derived from other columns in the dataset. For example:

        .. code-block:: python

           dataset = oc.open("haloproperties.hdf5")
           fof_halo_px = oc.col("fof_halo_mass")*oc.col("fof_halo_com_vx")

           dataset = dataset.select("fof_halo_mass", "*com*", fof_halo_px=fof_halo_px)

        This new dataset will contain the :code:`fof_halo_mass` columns, all the columns
        with :code:`com` in the center (e.g. :code:`fof_halo_com_vx`) and a new
        :code:`fof_halo_px` column.

        Parameters
        ----------
        *columns : str or list[str]
            The column or columns to select.

        **derived_columns : DerivedColumn
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
        all_columns: set[str] = set()
        for col_group in columns:
            if isinstance(col_group, str):
                col_group = {col_group}
            all_columns.update(col_group)

        new_state = self.__state
        if derived_columns:
            new_state = st.with_new_columns(new_state, {}, False, **derived_columns)
            all_columns.update(derived_columns.keys())

        return Dataset(st.select(new_state, all_columns))

    def drop(self, *columns: str | Iterable[str]) -> Dataset:
        """
        Create a new dataset without the provided columns. This
        function accepts wildcards. For exampe, "fof*" will drop all columns
        that start with "fof", while "*com*" will drop all columns that have
        "com" somewhere in the middle.

        Parameters
        ----------
        *columns : str or list[str]
            The columns to drop

        Returns
        -------
        dataset : Dataset
            The new dataset without the dropped columns

        Raises
        ------
        ValueError
            If any of the provided columns are not in the dataset.

        """

        all_columns: set[str] = set()
        for col_group in columns:
            if isinstance(col_group, str):
                col_group = {col_group}
            all_columns.update(col_group)

        return Dataset(st.select(self.__state, all_columns, drop=True))

    def sort_by(self, column: Optional[str], invert: bool = False) -> Dataset:
        """
        Sort this dataset by the values in a given column. By default sorting is in
        ascending order (least to greatest). Pass invert = True to sort in descending
        order (greatest to least).

        This can be used to, for example, select largest halos in a given
        dataset:

        .. code-block:: python

            dataset = oc.open("haloproperties.hdf5")
            dataset = dataset
                        .sort_by("fof_halo_mass", invert=True)
                        .take(100, at="start")

        Parameters
        ----------
        column : Optional[str]
            The column in the halo_properties or galaxy_properties dataset to
            order the collection by. Pass :code:`None` to remove sorting.

        invert : bool, default = False
            If False (the default), ordering will be from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : Dataset
            A new Dataset ordered by the given column.


        """
        return Dataset(st.sort_by(self.__state, column, invert))

    def take(
        self, n: int, at: str = "random", mode: Literal["local", "global"] = "local"
    ) -> Dataset:
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
        dataset : Dataset
            The new dataset with only the selected rows.

        Raises
        ------
        ValueError
            If n is negative or greater than the number of rows in the dataset,
            or if 'at' is invalid.

        """
        return Dataset(st.take(self.__state, n, at, mode))

    def take_range(
        self, start: int, end: int, mode: Literal["local", "global"] = "local"
    ) -> Dataset:
        """
        Create a new dataset from a row range in this dataset. We use standard
        indexing conventions, so the rows included will be start -> end - 1.

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
              of that range it owns. If the dataset is sorted, ranks will
              coordinate to take from the globally-sorted dataset.

        Returns
        -------
        dataset : Dataset
            The new dataset with only the rows from start to end.

        Raises
        ------
        ValueError
            If start or end are negative or greater than the length of the dataset
            or if end is greater than start.

        """
        return Dataset(st.take_range(self.__state, start, end, mode))

    def take_rows(self, rows: np.ndarray | DataIndex):
        """
        Take the rows of a dataset specified by the :code:`rows` argument.
        :code:`rows` should be an array of integers.

        Parameters:
        -----------
        rows: np.ndarray[int]

        Returns
        -------
        dataset: The dataset with only the specified rows included

        Raises:
        -------
        ValueError:
            If any of the indices is less than 0 or greater than the length of the
            dataset.

        """

        row_range = get_range(rows)
        if row_range[0] < 0 or row_range[1] > len(self):
            raise ValueError(
                "Row indices must be between 0 and the length of this dataset - 1!"
            )

        return Dataset(st.take_rows(self.__state, rows))

    def with_new_columns(
        self,
        descriptions: str | dict[str, str] = {},
        allow_overwrite: bool = False,
        **new_columns: ConstructedColumn | Column | np.ndarray | u.Quantity,
    ):
        """
        Create a new dataset with additional columns. These new columns can be derived
        from columns already in the dataset, a numpy array, or an astropy quantity
        array. When a column is derived from other columns, it will behave
        appropriately under unit transformations. Columns provided directly as astropy
        quantities will not change under unit transformations. See
        :ref:`Adding Custom Columns` for examples.

        If allow_overwrite is :code:`True`, the new column may have the same name as
        a column that already exists in the dataset. This can be used to transform a column,
        for example:

        .. code-block:: python

           log_mass = oc.col("fof_halo_mass").log10()
           ds = ds.with_new_columns(fof_halo_mass=log_mass, allow_overwrite=True)

        The "fof_halo_mass" column will now be the log of the original "fof_halo_mass" column.

        Columns will be given the same name as the argument you use when you pass them into the function.
        For example, we could do the same as above but name the column "log_fof_halo_mass" with

        .. code-block:: python

            log_mass = oc.col("fof_halo_mass").logo10()
            ds = ds.with_new_columns(log_fof_halo_mass = log_mass)

        Parameters
        ----------

        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`Dataset.descriptions <opencosmo.Dataset.descriptions>`. If a dictionary,
            should have keys matching the column names.
        allow_overwrites : bool, default = False
            If false, attempting to add a new column with the same name as an existing column will throw an error.
            If true, overwrites are allowed.

        ** new_columns : opencosmo.DerivedColumn | np.ndarray | units.Quantity
            The new columns to add. The name of the argument is the name the column will take.

        Returns
        -------
        dataset : opencosmo.Dataset
            This dataset with the columns added

        """
        if isinstance(descriptions, str):
            descriptions = {key: descriptions for key in new_columns.keys()}
        return Dataset(
            st.with_new_columns(
                self.__state, descriptions, allow_overwrite, **new_columns
            )
        )

    def make_schema(
        self, with_header: bool = True, name: Optional[str] = None
    ) -> Schema:
        """
        Prep to write the dataset. This should not be called directly for the user.
        The opencosmo.write file writer automatically handles the file context.

        Parameters
        ----------
        file : h5py.File
            The file to write to.
        dataset_name : str
            The name of the dataset in the file. The default is "data".

        """
        return st.make_schema(self.__state, name)

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns: u.Unit,
    ) -> Dataset:
        r"""
        Create a new dataset from this one with a different unit convention, and/or
        convert one unit to another across the entire dataset, or convert individual
        columns.

        Unit conversions are always performed after a change of convention, and
        changing conventions clears any existing unit conversions. Individual
        column conversions always take precedence over blanket unit conversions.

        Calling this function without arguments will clear any existing unit conversions.

        For more, see :doc:`units`.

        .. code-block:: python

            import astropy.units as u

            # this works
            dataset = dataset.with_units(fof_halo_mass=u.kg)

            # this clears the previous conversion
            dataset = dataset.with_units("scalefree")

            # This now fails, because the units of masses
            # are Msun / h, which cannot be converted to kg
            dataset = dataset.with_units(fof_halo_mass=u.kg)

            # this will work, the units of halo mass in the "physical"
            # convention are Msun (no h).
            dataset = dataset.with_units("physical", fof_halo_mass=u.kg, fof_halo_center_x=u.lyr)

            # Suppose you want all distances in lightyears, but the x coordinate of your
            # halo center in kilometers, for some reason ¯\_(ツ)_/¯
            blanket_conversions = {u.Mpc: u.lyr}
            dataset = dataset.with_units(conversions = blanket_conversions, fof_halo_center_x = u.km)



        Parameters
        ----------
        convention : str, optional
            The unit convention to use. One of "physical", "comoving",
            "scalefree", or "unitless".

        conversions : dict[astropy.units.Unit, astropy.Units.Unit]
            Conversions that apply to all columns in the dataset with the
            unit given by the key.

        **column_conversions: astropy.units.Unit
            Custom unit conversions for one or more or of the columns
            in this dataset.

        Returns
        -------
        dataset : Dataset
            The new dataset with the requested unit convention and/or conversions.

        """
        return Dataset(st.with_units(self.__state, convention, conversions, columns))
