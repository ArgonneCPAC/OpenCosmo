from __future__ import annotations

from functools import reduce
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Optional,
    TypeAlias,
)
from warnings import warn

import astropy.units as u  # type: ignore
import numpy as np
from astropy.table import QTable  # type: ignore

from opencosmo.dataset.evaluate import verify_for_lazy_evaluation, visit_dataset
from opencosmo.dataset.formats import convert_data, verify_format
from opencosmo.index import into_array, mask, project
from opencosmo.spatial import check
from opencosmo.units.converters import get_scale_factor

if TYPE_CHECKING:
    from astropy import units
    from astropy.cosmology import Cosmology

    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.dataset.state import DatasetState
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema
    from opencosmo.parameters import HaccSimulationParameters
    from opencosmo.spatial.protocols import Region
    from opencosmo.spatial.tree import Tree


OpenCosmoData: TypeAlias = QTable | u.Quantity | dict[str, np.ndarray] | np.ndarray


class Dataset:
    def __init__(
        self,
        header: OpenCosmoHeader,
        state: DatasetState,
        tree: Optional[Tree] = None,
    ):
        self.__header = header
        self.__state = state
        self.__tree = tree

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
        head = f"OpenCosmo Dataset (length={length})\n"
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        return head + cosmo_repr + table_head + table_repr

    @property
    def index(self):
        return self.__state.raw_index

    def __len__(self):
        return len(self.__state)

    def __enter__(self):
        # Need to write tests
        return self

    def __exit__(self, *exc_details):
        return self.__state.__exit__(*exc_details)

    def close(self):
        return self.__state.__exit__()

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
        return self.__header

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
    def meta_columns(self) -> list[str]:
        return self.__state.meta_columns

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
    def cosmology(self) -> Cosmology:
        """
        The cosmology of the simulation this dataset is drawn from as
        an astropy.cosmology.Cosmology object.

        Returns
        -------
        cosmology: astropy.cosmology.Cosmology
        """
        return self.__header.cosmology

    @property
    def dtype(self) -> str:
        """
        The data type of this dataset.

        Returns
        -------
        dtype: str
        """
        return str(self.__header.file.data_type)

    @property
    def redshift(self) -> float | tuple[float, float] | None:
        """
        The redshift slice or range this dataset was drawn from

        Returns
        -------
        redshift: float

        """
        return self.__header.file.redshift

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
    def simulation(self) -> HaccSimulationParameters:
        """
        The parameters of the simulation this dataset is drawn
        from.

        Returns
        -------
        parameters: opencosmo.parameters.hacc.HaccSimulationParameters
        """
        return self.__header.simulation

    @property
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
        # should rename this, dataset.data can get confusing
        # Also the point is that there's MORE data than just the table
        return self.get_data("astropy")

    def get_metadata(self, columns: str | list[str] = []):
        if isinstance(columns, str):
            columns = [columns]

        return self.__state.get_metadata(columns)

    def get_data(
        self,
        output="astropy",
        unpack=True,
        metadata_columns=[],
    ) -> OpenCosmoData:
        """
        Get the data in this dataset as an astropy table/column or as
        numpy array(s). Note that a dataset does not load data from disk into
        memory until this function is called. As a result, you should not call
        this function until you have performed any transformations you plan to
        on the data.

        The method supports output into several different formats, including
        "astropy", "numpy", "pandas", "polars", and "pyarrow". Although astropy
        and numpy are core dependencies of OpenCosmo, the remaining formats
        require you to have the relevant libraries installed in your python
        environment. This method will check that it can import the necessary
        libraries before attempting to read data. Note that outputting as
        "polars" or "arrow" requires copying the data out of its original
        numpy arrays, which will impact performance.

        If the dataset only contains a single column, it will not be put in a table
        or dictionary. "astropy", "numpy" and "arrow" will return a single array
        in this case, while "polars" and "pandas" will return a Series object.

        Parameters
        ----------
        output: str, default="astropy"
            The format to output the data in.
            Currently supported are "astropy", "numpy", "pandas", "polars", "arrow"

        Returns
        -------
        data: Any
            The data in this dataset.
        """
        verify_format(output)

        if self.__state.convention.value == "physical":
            scale_factor = get_scale_factor(self.__state, self.cosmology, self.redshift)
            unit_kwargs = {"scale_factor": scale_factor}
        else:
            unit_kwargs = {}

        data = self.__state.get_data(
            unit_kwargs=unit_kwargs, metadata_columns=metadata_columns
        )  # dict
        if unpack:
            data = {
                key: value[0]
                if isinstance(value, np.ndarray) and len(value) == 1
                else value
                for key, value in data.items()
            }

        return convert_data(data, output)

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

        if self.__tree is None:
            raise AttributeError(
                "Your dataset does not contain a spatial index, "
                "so spatial querying is not available"
            )

        if not self.header.file.is_lightcone:
            columns = check.find_coordinates_3d(self, self.dtype)

            check_region = region.into_base_convention(
                self.__state.unit_handler,
                columns,
                self.__state.convention,
                {
                    "scale_factor": self.cosmology.scale_factor(
                        self.header.file.redshift
                    ).value
                },
            )
        else:
            check_region = region

        if not self.__state.region.intersects(check_region):
            new_state = self.__state.take_rows(np.array([]))
            return Dataset(self.__header, new_state, self.__tree)

        if not self.__state.region.contains(check_region):
            warn(
                "You're querying with a region that is not fully contained by the "
                "region this dataset is in. This may result in unexpected behavior"
            )

        contained_index: DataIndex
        intersects_index: DataIndex
        contained_index, intersects_index = self.__tree.query(check_region)

        contained_index = project(self.__state.raw_index, contained_index)
        intersects_index = project(self.__state.raw_index, intersects_index)

        check_state = self.__state.take_rows(intersects_index)
        check_dataset = Dataset(
            self.__header,
            check_state,
            self.__tree,
        )
        if not self.__header.file.is_lightcone:
            check_dataset = check_dataset.with_units("scalefree")

        index_mask = check.check_containment(
            check_dataset, check_region, self.__header.file
        )
        new_intersects_index = mask(intersects_index, index_mask)

        new_index = np.concatenate(
            [into_array(contained_index), into_array(new_intersects_index)]
        )

        new_state = self.__state.take_rows(new_index).with_region(check_region)

        return Dataset(self.__header, new_state, self.__tree)

    def evaluate(
        self,
        func: Callable,
        vectorize=False,
        insert=True,
        format="astropy",
        **evaluate_kwargs,
    ) -> Dataset | np.ndarray:
        """
        Iterate over the rows in this dataset, apply :code:`func` to each, and collect
        the result as new columns in the dataset.

        This function is the equivalent of :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`
        for cases where the new column is not a simple algebraic combination of existing columns. Unlike
        :code:`with_new_columns`, this method will evaluate the results immediately and the resulting
        columns will not change under unit transformations. You may also choose to simply return the result
        instead of adding it as a column.

        The function should take in arguments with the same name as the columns in this dataset that
        are needed for the computation, and should return a dictionary of output values.
        The dataset will automatically selected the needed columns to avoid reading unnecessarily reading
        data from disk

        The new columns will have the same names as the keys of the output dictionary
        See :ref:`Evaluating On Datasets` for more details.

        If vectorize is set to True, the full columns will be pased to the dataset. Otherwise,
        rows will be passed to the function one at a time.

        If the function returns None, this method will also return None as output. For example, the function
        could simply produce plots and save the to files.

        Parameters
        ----------

        func: Callable
            The function to evaluate on the rows in the dataset.

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False)

        insert: bool, default = True
            If true, the data will be inserted as a column in this dataset. The new column will have the same name
            as the function. Otherwise the data will be returned directly.

        format: str, default = astropy
            Whether to provide data to your function as "astropy" quantities or "numpy" arrays/scalars. Default "astropy". Note that
            this method does not support all the formats available in :py:meth:`get_data <opencosmo.Dataset.get_data>`

        **evaluate_kwargs: any,
            Any additional arguments that are required for your function to run. These will be passed directly
            to the function as keyword arguments. If a kwarg is an array of values with the same length as the dataset,
            it will be treated as an additional column.

        Returns
        -------
        result : Dataset | dict[str, np.ndarray | astropy.units.Quantity]
            The new dataset with the evaluated column(s) or the results as numpy arrays or astropy quantities
        """
        if format not in ["astropy", "numpy"]:
            raise ValueError(
                f"Evaluate only supports numpy and astropy format, got: {format}"
            )
        kwarg_columns = set(evaluate_kwargs.keys()).intersection(self.columns)
        if kwarg_columns:
            raise ValueError(
                "Keyword arguments cannot have the same name as columns in your dataset!"
            )
        strategy = evaluate_kwargs.pop(
            "strategy", "row_wise" if not vectorize else "vectorize"
        )
        if not insert:
            output = visit_dataset(func, strategy, format, evaluate_kwargs, self)
            return output

        evaluated_column = verify_for_lazy_evaluation(
            func, strategy, format, evaluate_kwargs, self
        )
        return self.with_new_columns(
            descriptions={}, **{func.__name__: evaluated_column}
        )

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
        if not masks:
            return self
        required_columns: set[str] = reduce(
            lambda acc, r: acc | r.requires, masks, set()
        )
        data = self.select(required_columns).get_data(unpack=False)
        bool_mask = np.ones(len(data), dtype=bool)
        for m in masks:
            bool_mask &= m.apply(data)

        new_state = self.__state.with_mask(bool_mask)
        return Dataset(self.__header, new_state, self.__tree)

    def rows(
        self,
        include_units: bool = True,
        metadata_columns=[],
    ) -> Generator[Mapping[str, float | units.Quantity | np.ndarray]]:
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
        if self.__state.convention.value == "physical":
            scale_factor = get_scale_factor(self.__state, self.cosmology, self.redshift)
            unit_kwargs = {"scale_factor": scale_factor}
        else:
            unit_kwargs = {}

        for row in self.__state.rows(metadata_columns, unit_kwargs):
            output_data = row
            if not isinstance(output_data, dict):
                output_data = {self.columns[0]: row}

            if not include_units:
                output_data = {
                    name: val.value if isinstance(val, u.Quantity) else val
                    for name, val in output_data.items()
                }
            yield output_data

    def select(self, columns: str | Iterable[str]) -> Dataset:
        """
        Create a new dataset from a subset of columns in this dataset

        Parameters
        ----------
        columns : str or list[str]
            The column or columns to select.

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected columns.

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        new_state = self.__state.select(columns)
        return Dataset(
            self.__header,
            new_state,
            self.__tree,
        )

    def drop(self, columns: str | Iterable[str]) -> Dataset:
        """
        Create a new dataset without the provided columns.

        Parameters
        ----------
        columns : str or list[str]
            The columns to drop

        Returns
        -------
        dataset : Dataset
            The new dataset without the droppedcolumns

        Raises
        ------
        ValueError
            If any of the provided columns are not in the dataset.

        """
        if isinstance(columns, str):
            columns = [columns]

        current_columns = set(self.__state.columns)
        dropped_columns = set(columns)

        if missing := dropped_columns.difference(current_columns):
            raise ValueError(f"Columns {missing} are  not in this dataset")

        return self.select(current_columns - dropped_columns)

    def sort_by(self, column: str, invert: bool = False) -> Dataset:
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
        column : str
            The column in the halo_properties or galaxy_properties dataset to
            order the collection by.

        invert : bool, default = False
            If False (the default), ordering will be from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : Dataset
            A new Dataset ordered by the given column.


        """
        new_state = self.__state.sort_by(column, invert)
        return Dataset(
            self.__header,
            new_state,
            self.__tree,
        )

    def take(
        self,
        n: int,
        at: str = "random",
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

        new_state = self.__state.take(n, at)

        return Dataset(
            self.__header,
            new_state,
            self.__tree,
        )

    def take_range(self, start: int, end: int) -> Dataset:
        """
        Create a new dataset from a row range in this dataset. We use standard
        indexing conventions, so the rows included will be start -> end - 1.

        Parameters
        ----------
        start : int
            The beginning of the range
        end : int
            The end of the range

        Returns
        -------
        table : astropy.table.Table
            The table with only the rows from start to end.

        Raises
        ------
        ValueError
            If start or end are negative or greater than the length of the dataset
            or if end is greater than start.

        """

        new_state = self.__state.take_range(start, end)

        return Dataset(
            self.__header,
            new_state,
            self.__tree,
        )

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
        new_state = self.__state.take_rows(rows)
        return Dataset(self.__header, new_state, self.__tree)

    def with_new_columns(
        self,
        descriptions: str | dict[str, str] = {},
        **new_columns: ConstructedColumn | np.ndarray | units.Quantity,
    ):
        """
        Create a new dataset with additional columns. These new columns can be derived
        from columns already in the dataset, a numpy array, or an astropy quantity
        array. When a column is derived from other columns, it will behave
        appropriately under unit transformations. Columns provided directly as astropy
        quantities will not change under unit transformations. See
        :ref:`Adding Custom Columns` for examples.

        Parameters
        ----------

        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`Dataset.descriptions <opencosmo.Dataset.descriptions>`. If a dictionary,
            should have keys matching the column names.

        ** new_columns : opencosmo.DerivedColumn | np.ndarray | units.Quantity

        Returns
        -------
        dataset : opencosmo.Dataset
            This dataset with the columns added

        """
        if isinstance(descriptions, str):
            descriptions = {key: descriptions for key in new_columns.keys()}
        new_state = self.__state.with_new_columns(descriptions, **new_columns)
        return Dataset(self.__header, new_state, self.__tree)

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
        schema = self.__state.make_schema(name)
        if self.__tree is not None:
            tree = self.__tree.apply_index(self.__state.raw_index)
            tree_schema = tree.make_schema()
            schema.children["index"] = tree_schema
        metadata = self.header.dump()
        schema.children["header"] = metadata

        return schema

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

        new_state = self.__state.with_units(
            convention, conversions, columns, self.cosmology, self.redshift
        )
        if convention is not None:
            new_header = self.__header.with_units(convention)
        else:
            new_header = self.__header

        return Dataset(
            new_header,
            new_state,
            self.__tree,
        )
