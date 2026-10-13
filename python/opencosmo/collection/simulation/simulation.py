from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Literal, Optional

from opencosmo import facade
from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.simulation import operations as simops
from opencosmo.collection.simulation import state as simst
from opencosmo.collection.structure import structure as sc
from opencosmo.collection.structure.state import StructureState
from opencosmo.dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import KeysView, Mapping

    import astropy.units as u
    import numpy as np

    from opencosmo.collection.protocols import Collection
    from opencosmo.collection.simulation.state import SimulationChild
    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.dataset.dataset import OpenCosmoData
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.mapping.mapping import DatasetMatchSet
    from opencosmo.spatial.protocols import Region

    UserChild = Dataset | lc.Lightcone | sc.StructureCollection


class SimulationCollection:
    """
    A collection of datasets of the same type from different
    simulations. In general this exposes the exact same API
    as the individual datasets, but maps the results across
    all of them.
    """

    def __init__(self, state: simst.SimulationState):
        self.__state = state

    @classmethod
    def from_datasets(
        cls,
        datasets: Mapping[str, Dataset | lc.Lightcone | Collection],
        match_set: DatasetMatchSet | None = None,
    ) -> SimulationCollection:
        """
        Create a simulation collection from datasets, lightcones, or structure
        collections drawn from different simulations.

        Parameters
        ----------
        datasets: Mapping[str, Dataset | Lightcone | StructureCollection]
            The members of the collection, keyed by simulation name.

        match_set: DatasetMatchSet, optional
            Row matching between the datasets. Only supported when every member
            is a :py:class:`Dataset <opencosmo.Dataset>`.

        Returns
        -------
        collection: opencosmo.SimulationCollection

        Raises
        ------
        ValueError
            If a member is not a dataset, lightcone, or structure collection, or
            a match set is given for members that are not all datasets.
        """
        members: dict[str, SimulationChild] = {}
        for name, value in datasets.items():
            if not isinstance(value, (Dataset, lc.Lightcone, sc.StructureCollection)):
                raise ValueError(
                    "Simulation collection only accepts datasets, lightcones, "
                    "and structure collections"
                )
            members[name] = value._state
        return cls(simst.from_parts(members, match_set))

    @property
    def _state(self) -> simst.SimulationState:
        """Return the internal state for collection and I/O implementations."""
        return self.__state

    def __with_state(self, state: simst.SimulationState) -> SimulationCollection:
        if state is self.__state:
            return self
        return SimulationCollection(state)

    def __getattr__(self, key: str):
        if key == "_SimulationCollection__state":
            raise AttributeError(key)
        try:
            return {
                name: getattr(header, key)
                for name, header in self.__state.header.items()
            }
        except AttributeError:
            return object.__getattribute__(self, key)

    def __dir__(self) -> list[str]:
        keys: set[str] = set()
        for header in self.__state.header.values():
            keys.update(header.parameters.keys())
        return list(keys) + list(super().__dir__())

    def keys(self) -> KeysView[str]:
        return dict.fromkeys(simst.public_keys(self.__state)).keys()

    def values(self):
        return [facade.wrap(child) for child in self.__state.children.values()]

    def items(self):
        return {
            name: facade.wrap(child) for name, child in self.__state.children.items()
        }.items()

    def __len__(self):
        return len(self.__state)

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key: str):
        return facade.wrap(simst.view(self.__state, key))

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for child in self.values():
            child.__exit__(*exc_details)

    def __repr__(self):
        members = self.__state.members.values()
        n_collections = sum(1 for v in members if isinstance(v, StructureState))
        n_datasets = len(members) - n_collections
        return (
            f"SimulationCollection({n_collections} collections, {n_datasets} datasets)"
        )

    @classmethod
    def open(cls, targets: list[DatasetTarget], **kwargs) -> Collection | Dataset:
        raise NotImplementedError()

    def make_schema(self, path: str) -> Schema:
        return simops.make_schema(self.__state, path)

    @property
    def dtype(self) -> dict[str, str]:
        return self.__state.dtype

    @property
    def header(self) -> dict[str, OpenCosmoHeader]:
        return self.__state.header

    def get_data(
        self, format: str = "astropy", wrap_single: bool = False, unpack: bool = True
    ) -> dict[str, OpenCosmoData]:
        """
        Retrieve the data from all datasets in this collection, and return it as a dictionary of
        type (dataset_name, data). This method requires that the underlying entries into this
        SimulationCollection are :py:class:`datasets <opencosmo.Dataset>` or :py:class:`lightcones <opencosmo.Lightcone>`,
        other entries will raise an error.

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
        data: dict
            The data the datasets. Keys will be the same as the keys of this
            collection.

        """
        return simops.get_data(self.__state, format, unpack, wrap_single)

    def match(self, dataset: str) -> SimulationCollection:
        """
        Create a new simulation collection where the datasets are ordered so that matched
        objects appear in the same row across every dataset. All datasets are matched
        to `source`, and only rows that are available in every simulation are included.
        This feature requires that the underlying data has a pre-built matching index,
        and will raise an error if the index is not present.

        For example, suppose you have one gravity-only sim and one hydro sim.

        .. code-block:: python

            collection = collection.match("gravity_only")

        Further operations such as :py:meth:`take <opencosmo.SimulationCollection.take>` and
        :py:meth:`filter <opencosmo.SimulationCollection.filter>` will now operate exclusively
        on the gravity_only sim, and retain matching objects in the other sim. For a matched
        collection, calling :py:meth:`get_data <opencosmo.SimulationCollection.get_data>` will
        always return datasets with equal length

        To clear a match, use :py:meth:`clear_match <opencosmo.SimulationCollection.clear_match>`.
        Clearning a match does not undo any previous operations, it simple causes later
        operations to be applied to all datasets regardless of matching between rows.

        Parameters
        ----------
        dataset: str
            The dataset to match the collection against. You can get the available
            names with SimulationCollection.keys()

        Returns
        -------
        dataset: opencosmo.SimulationCollection
            The collection with the matching performed.

        Raises
        -------
        ValueError: If the data does not contain a matching index
        """

        return self.__with_state(simops.match(self.__state, dataset))

    def clear_match(self):
        """
        Clear a previously set :py:meth`match <opencosmo.SimulationCollection.match>`. This does not
        alter any datasets. Instead, it ensures that any future operations will be applied uniformly
        across all datasets, rather than only to the match source.

        Take the following example:

        .. code-block:: python

            collection = collection.match("gravity_only")
            collection = collection.filter(oc.col("fof_halo_mass") > 1e14)
            collection = collection.clear_match()

        This collection will contain all objects in the gravity-only simulations with
        a mass above 10**14 and their matched objects in the other simulations. The
        clear_match call has no effect until a further operation is performed. If you
        retrieve data, the rows in each dataset will be matched up.

        Another example:

        .. code-block:: python

            collection = collection.match("gravity_only")
            collection = collection.clear_match()
            collection = collection.filter(oc.col("fof_halo_mass") > 1e14)

        This collection will contain all objects which have any a match to an object in the
        gravity-only simulation, and is above a mass 10**14. However since the filter
        was performed after clear_match, the order of the datasets are no longer
        guaranteed and the datasets may be of different lengths.

        If the collection does not contain a matching index, or no matching has been performed,
        this operations has no effect.

        Returns
        -------
        dataset: opencosmo.SimulationCollection
            The collection with the matching cleared.
        """

        return self.__with_state(simops.clear_match(self.__state))

    def bound(
        self, region: Region, select_by: Optional[str] = None
    ) -> SimulationCollection:
        """
        Restrict the datasets to some region. Note that the SimulationCollection does
        not do any checking to ensure its members have identical boxes. As a result
        this method can in principle fail for some of the simulations in the
        collection and not others. This should never happen when working with official
        OpenCosmo data products.

        See :doc:`spatial_ref` for details of how to construct regions.

        Parameters
        ----------
        region: opencosmo.spatial.Region
            The region to query

        Returns
        -------
        dataset: opencosmo.SimulationCollection
            The portion of each dataset inside the selected region

        """
        return self.__with_state(simops.bound(self.__state, region, select_by))

    def filter(self, *masks: ColumnMask, **kwargs) -> SimulationCollection:
        """
        Filter the datasets in the collection. This method behaves
        exactly like :meth:`opencosmo.Dataset.filter` or
        :meth:`opencosmo.StructureCollection.filter`, but
        it applies the filter to all the datasets or collections
        within this collection. The result is a new collection.

        If this dataset has been matched with :py:meth:`SimulationCollection.match <opencosmo.SimulatinCollection.match>`,
        the filter will only run on the dataset matched against, with all matching rows retained
        in the remaining datasets.

        Parameters
        ----------
        filters:
            The filters constructed with :func:`opencosmo.col`

        Returns
        -------
        SimulationCollection
            A new collection with the same datasets, but only the
            particles that pass the filter.
        """
        return self.__with_state(simops.filter(self.__state, *masks, **kwargs))

    def select(self, *args, **kwargs) -> SimulationCollection:
        """
        Select a set of columns in the datasets in this collection. This method
        calls the underlying method in :class:`opencosmo.Dataset`, or
        :class:`opencosmo.StructureCollection` depending on the context. As such
        its behavior and arguments can vary depending on what this collection
        contains. See the documentation for those objects to determine
        the expected arguments.

        If the collection holds datasets with different column sets (e.g a matched
        gravity-only and hydro sim) it will make a best-effort attempt to distribute
        the selections to the relevant dataset. Datasets with no matches will
        be retained unchanged.


        Parameters
        ----------
        args:
            The arguments to pass to the select method. This is
            usually a list of column names to select.
        kwargs:
            The keyword arguments to pass to the select method.
            This is usually a dictionary of column names to select.

        Returns
        -------
        SimulationCollection
            A new collection with only the specified columns

        """
        return self.__with_state(simops.select(self.__state, *args, **kwargs))

    def drop(self, *args, **kwargs) -> SimulationCollection:
        """
        Drop columns by automatically matching their names to datasets in this
        collection. Wildcards are applied to every dataset, while datasets
        without a match are unchanged. For nested collections, matching follows
        their :meth:`drop` behavior.

        To target datasets explicitly, pass dataset names as keyword arguments.
        This form is forwarded to the underlying datasets or collections.

        If the collection holds datasets with different column sets (e.g a matched
        gravity-only and hydro sim) it will make a best-effort attempt to distribute
        the selections to the relevant dataset. Datasets with no matches will
        be retained unchanged.


        Parameters
        ----------
        args : str or Iterable[str]
            Column names or wildcard patterns to match automatically across all
            datasets in the collection.
        kwargs
            Explicit dataset-keyed drop selections.

        """
        return self.__with_state(simops.drop(self.__state, *args, **kwargs))

    def take(
        self, n: int, at: str = "random", mode: Literal["local", "global"] = "local"
    ) -> SimulationCollection:
        """
        Take a subest of rows from all datasets or collections in this collection.
        This method will delegate to the underlying method in
        :class:`opencosmo.Dataset`, or :class:`opencosmo.StructureCollection` depending
        on  the context. As such, behavior may vary depending on what this collection
        contains. See their documentation for more info.

        If this dataset has been matched with :py:meth:`SimulationCollection.match <opencosmo.SimulatinCollection.match>`,
        the take operation will only run on the dataset matched against, with all matching rows retained
        in the remaining datasets.


        Parameters
        ----------
        n: int
            The number of rows to take
        at: str, default = "random"
            The method to use to take rows. Must be one of "start", "end", "random".

        """
        return self.__with_state(simops.take(self.__state, n, at, mode))

    def take_range(
        self, start: int, end: int, mode: Literal["local", "global"] = "local"
    ):
        """
        Take a range of rows from all datasets or collections in this collection.
        This method will fail if :code:`start` < 0, or any of the datasets are not at least
        :code:`end` long.

        If this dataset has been matched with :py:meth:`SimulationCollection.match <opencosmo.SimulatinCollection.match>`,
        the take operation will only run on the dataset matched against, with all matching rows retained
        in the remaining datasets.

        Parameters
        ----------
        n: int
            The number of rows to take
        at: str, default = "random"
            The method to use to take rows. Must be one of "start", "end", "random".

        Returns
        -------
        SimulationCollection
            The new simulation collection with only the specified rows.

        """
        return self.__with_state(simops.take_range(self.__state, start, end, mode))

    def take_rows(self, rows: np.ndarray | DataIndex) -> SimulationCollection:
        """
        Take the rows specified by :code:`rows` from all datasets or collections
        in this collection. :code:`rows` should be an array of integers, and is
        applied to every member.

        If this collection has been matched with :py:meth:`SimulationCollection.match <opencosmo.SimulationCollection.match>`,
        the rows are taken only from the dataset matched against, with all matching rows retained
        in the remaining datasets.

        Parameters
        ----------
        rows: np.ndarray[int]
            The rows to take

        Returns
        -------
        SimulationCollection
            The new simulation collection with only the specified rows.

        Raises
        ------
        ValueError
            If any of the indices is less than 0 or greater than the length of a
            dataset.
        """
        return self.__with_state(simops.take_rows(self.__state, rows))

    def with_new_columns(
        self,
        *args,
        datasets: Optional[str | Iterable[str]] = None,
        descriptions: str | dict[str, str] = {},
        allow_overwrite: bool = False,
        **new_columns: ConstructedColumn | np.ndarray,
    ):
        """
        Update the datasets within this collection with a set of new columns.
        This method simply calls :py:meth:`opencosmo.Dataset.with_new_columns` or
        :py:meth:`opencosmo.StructureCollection.with_new_columns`, as appropriate.

        You can also optionally pass the "datasets" keyword argument to specify that the
        operation should only be performed on a subset of the datasets.

        If passing in numpy arrays or astropy quantities, they should be provided
        as a dictionary where the keys are the same as the keys in this dataset.

        Parameters
        ----------
        datasets: str | list[str], optional
            The datasets to add the columns to.

        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`SimulationCollection(datasets).descriptions <opencosmo.SimulationCollection.descriptions>`.
            If a dictionary, should have keys matching the column names.

        allow_overwrite: bool, default = False


        ** columns : opencosmo.Column | np.ndarray | units.Quantity
            The new columns
        """
        return self.__with_state(
            simops.with_new_columns(
                self.__state,
                *args,
                datasets=datasets,
                descriptions=descriptions,
                allow_overwrite=allow_overwrite,
                **new_columns,
            )
        )

    def evaluate(
        self,
        func: Callable,
        datasets: Optional[str | Iterable[str]] = None,
        format: str = "astropy",
        vectorize: bool = False,
        insert: bool = False,
        allow_overwrite: bool = False,
        **evaluate_kwargs,
    ):
        """
        Evaluate the function :code:`func` on each of the datasets or collections
        held by this SimulationCollection. This function simply delegates to the
        either :py:meth:`StructureCollection.evaluate <opencosmo.StructureCollection.Evaluate>`
        or :py:meth:`Dataset.evaluate <opencosmo.Dataset.Evaluate>` as appropriate. Refer
        to :ref:`Evaluating Complex Expressions on Datasets and Collections` for more details.

        If "datasets" is provided, the evaluation will only be performed on the provided
        datasets.

        Parameters
        ----------

        func: Callable
            The function to evaluate
        datasets: str | list[str], optional
            The datasets to evaluate on. If not provided, will be evaluated on all datasets
        format: str, default = "astropy"
            The format in which to provide column data to your function. Supports the same formats
            as :py:meth:`get_data <opencosmo.Dataset.get_data>` ("astropy", "numpy", "pandas",
            "polars", "arrow", "jax"). When :code:`insert=True`, the function's output is converted
            back to numpy before being stored.

        vectorize: bool, default = False
            Whether to vectorize the computation. See :py:meth:`StructureCollection.evaluate <opencosmo.StructureCollection.Evaluate>`
            and/or :py:meth:`Dataset.evaluate <opencosmo.Dataset.Evaluate>` for more details.
        insert: bool, default = True
            Whether or not to insert the results as columns in the datasets. If false, the results will
            be returned directly. If true, this method will return a new Simulation Collection.

        Returns
        -------
        results: SimulationCollection | dict[str, np.ndarray] | dict[str, astropy.units.Quantity]
            The results of the computation, or a new simulation collection with the results inserted.
        """
        result = simops.evaluate(
            self.__state,
            func,
            datasets,
            format,
            vectorize,
            insert,
            allow_overwrite,
            **evaluate_kwargs,
        )
        if isinstance(result, simst.SimulationState):
            return self.__with_state(result)
        if next(iter(result.values())) is None:
            return None
        return result

    def sort_by(self, column: str, invert: bool = False):
        """
        Re-order the individual datasets in the collection based on a column. See
        :py:meth:`Dataset.sort_by <opencosmo.Dataset.sort_by>` for usage details.

        Parameters
        ----------
        column : str
            The column in the halo_properties or galaxy_properties dataset to
            order the collection by.

        invert : bool, default = False
            If False (the default) ordering will be done from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : SimulationCollection
            A new SimulationCollection with the datasets ordered by the given column.

        """
        return self.__with_state(simops.sort_by(self.__state, column, invert))

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns: u.Unit,
    ) -> SimulationCollection:
        """
        Transform all datasets or collections to use the given unit convention, convert
        all columns with a given unit into a different unit, and/or convert specific column(s)
        to a compatible unit. This method behaves exactly like :meth:`opencosmo.Dataset.with_units`.

        Parameters
        ----------
        convention: str
            The unit convention to use. One of "unitless",
            "scalefree", "comoving", or "physical".

        conversions: dict[astropy.units.Unit, astropy.units.Unit]
            Conversions that apply to all columns in the collection with the
            unit given by the key.

        **column_conversions: astropy.units.Unit
            Custom unit conversions for any column with a specific
            name in the datasets in this collection.

        Returns
        -------
        collection
            A new simulation collection with the requested unit conventions and conversions.


        """
        return self.__with_state(
            simops.with_units(self.__state, convention, conversions, **columns)
        )
