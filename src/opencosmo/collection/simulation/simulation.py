from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Optional, Self

from opencosmo.collection import structure as sc
from opencosmo.dataset import Dataset
from opencosmo.io import io
from opencosmo.io.schema import FileEntry, make_schema

if TYPE_CHECKING:
    import astropy.units as u
    import h5py
    from astropy.cosmology import Cosmology

    from opencosmo.collection.protocols import Collection
    from opencosmo.column.column import ColumnMask
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.schema import Schema
    from opencosmo.parameters import HaccSimulationParameters
    from opencosmo.spatial.protocols import Region


def verify_datasets_exist(file: h5py.File, datasets: Iterable[str]):
    """
    Verify a set of datasets exist in a given file.
    """
    if not set(datasets).issubset(set(file.keys())):
        raise ValueError(f"Some of {', '.join(datasets)} not found in file.")


class SimulationCollection(dict):
    """
    A collection of datasets of the same type from different
    simulations. In general this exposes the exact same API
    as the individual datasets, but maps the results across
    all of them.
    """

    def __init__(self, datasets: Mapping[str, Dataset | Collection]):
        self.update(datasets)

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for dataset in self.values():
            try:
                dataset.close()
            except ValueError:
                continue

    def __repr__(self):
        n_collections = sum(
            1
            for v in self.values()
            if isinstance(v, (SimulationCollection, sc.StructureCollection))
        )
        n_datasets = sum(1 for v in self.values() if isinstance(v, Dataset))
        return (
            f"SimulationCollection({n_collections} collections, {n_datasets} datasets)"
        )

    @classmethod
    def open(cls, targets: list[io.OpenTarget], **kwargs) -> Collection | Dataset:
        targets_by_name = {
            target.group.name.split("/")[-1]: target for target in targets
        }
        if len(targets_by_name) != len(targets):
            raise ValueError(
                "Not all datasets in this SimulationCollection have unique names!"
            )

        datasets = {
            name: io.open_single_dataset(target)
            for name, target in targets_by_name.items()
        }

        if len(datasets) == 1:
            return next(iter(datasets.values()))
        return cls(datasets)

    def make_schema(self) -> Schema:
        children = {}

        for name, dataset in self.items():
            children[name] = dataset.make_schema()
        return make_schema("/", FileEntry.SIMULATION_COLLECTION, children=children)

    def __map(
        self,
        method,
        *args,
        construct=True,
        datasets: Optional[str | Iterable[str]] = None,
        **kwargs,
    ):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        regular_kwargs = {}
        mapped_kwargs = {}
        if isinstance(datasets, str):
            datasets = [datasets]
        elif datasets is None:
            datasets = self.keys()
        requested_datasets = set(datasets)
        if not requested_datasets.issubset(self.keys()):
            raise ValueError(
                f"Unknown datasets {requested_datasets.difference(self.keys())}"
            )

        for name, value in kwargs.items():
            if isinstance(value, dict) and set(value.keys()) == requested_datasets:
                mapped_kwargs[name] = value
            else:
                regular_kwargs[name] = value

        output = {}
        for name in requested_datasets:
            dataset = self[name]
            dataset_mapped_kwargs = {key: kw[name] for key, kw in mapped_kwargs.items()}
            output[name] = getattr(dataset, method)(
                *args, **regular_kwargs, **dataset_mapped_kwargs
            )
        if construct:
            return SimulationCollection(output)
        return output

    def __map_attribute(self, attribute):
        return {k: getattr(v, attribute) for k, v in self.items()}

    @property
    def dtype(self) -> dict[str, str]:
        return {key: ds.header.file.data_dtype for key, ds in self.items()}

    @property
    def header(self) -> dict[str, OpenCosmoHeader]:
        return self.__map_attribute("header")

    @property
    def cosmology(self) -> dict[str, Cosmology]:
        """
        Get the cosmologies of the simulations in the collection

        Returns
        --------
        cosmologies: dict[str, astropy.cosmology.Cosmology]
        """
        return self.__map_attribute("cosmology")

    @property
    def redshift(self) -> dict[str, float | tuple[float, float]]:
        """
        Get the redshift slices or ranges for the simulations in the collection

        Returns
        --------
        redshifts: dict[str, float | tuple[float,float]]
        """
        return self.__map_attribute("redshift")

    @property
    def simulation(self) -> dict[str, HaccSimulationParameters]:
        """
        Get the simulation parameters for the simulations in the collection

        Returns
        --------
        simulation_parameters: dict[str, opencosmo.parameters.HaccSimulationParameters]
        """

        return self.__map_attribute("simulation")

    def bound(self, region: Region, select_by: Optional[str] = None) -> Self:
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
        return self.__map("bound", region, select_by)

    def filter(self, *masks: ColumnMask, **kwargs) -> Self:
        """
        Filter the datasets in the collection. This method behaves
        exactly like :meth:`opencosmo.Dataset.filter` or
        :meth:`opencosmo.StructureCollection.filter`, but
        it applies the filter to all the datasets or collections
        within this collection. The result is a new collection.

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
        return self.__map("filter", *masks, **kwargs)

    def select(self, *args, **kwargs) -> Self:
        """
        Select a set of columns in the datasets in this collection. This method
        calls the underlying method in :class:`opencosmo.Dataset`, or
        :class:`opencosmo.Collection` depending on the context. As such
        its behavior and arguments can vary depending on what this collection
        contains.

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
        return self.__map("select", *args, **kwargs)

    def drop(self, *args, **kwargs) -> Self:
        """
        Drop a set of columns from the datasets in the collection. This method
        calls the underlying method in :class:`opencosmo.Dataset`, or
        :class:`opencosmo.Collection` depending on the context. As such
        its behavior and arguments can vary depending on what this collection
        contains.

        Parameters
        ----------
        args:
            The arguments to pass to the select method. This is
            usually a list of column names to drop.
        kwargs:
            The keyword arguments to pass to the select method.
            This is usually a dictionary of column names to select.

        """
        return self.__map("drop", *args, **kwargs)

    def take(self, n: int, at: str = "random") -> Self:
        """
        Take a subest of rows from all datasets or collections in this collection.
        This method will delegate to the underlying method in
        :class:`opencosmo.Dataset`, or :class:`opencosmo.StructureCollection` depending
        on  the context. As such, behavior may vary depending on what this collection
        contains. See their documentation for more info.

        Parameters
        ----------
        n: int
            The number of rows to take
        at: str, default = "random"
            The method to use to take rows. Must be one of "start", "end", "random".

        """
        if any(len(ds) < n for ds in self.values()):
            raise ValueError(
                f"Not all datasets in this collection have at least {n} rows!"
            )
        return self.__map("take", n, at)

    def take_range(self, start: int, end: int):
        """
        Take a range of rows from all datasets or collections in this collection.
        This method will fail if :code:`start` < 0, or any of the datasets are not at least
        :code:`end` long.

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
        if start < 0 or any(len(ds) < end for ds in self.values()):
            raise ValueError(
                "The range must be between zero and the length of the shortest dataset"
            )
        return self.__map("take_range", start, end)

    def with_new_columns(
        self,
        *args,
        datasets: Optional[str | Iterable[str]] = None,
        descriptions: str | dict[str, str] = {},
        **kwargs,
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

        ** columns : opencosmo.DerivedColumn | np.ndarray | units.Quantity
            The new columns
        """
        if datasets is not None:
            if isinstance(datasets, str):
                datasets = [datasets]
            else:
                datasets = list(datasets)

            output = {name: ds for name, ds in self.items()}
            for ds_name in datasets:
                output[ds_name] = output[ds_name].with_new_columns(
                    *args, descriptions=descriptions, **kwargs
                )
            return SimulationCollection(output)

        return self.__map(
            "with_new_columns", *args, descriptions=descriptions, **kwargs
        )

    def evaluate(
        self,
        func: Callable,
        datasets: Optional[str | Iterable[str]] = None,
        format: str = "astropy",
        vectorize: bool = False,
        insert: bool = False,
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
            The format of the data that is provided to your function. If "astropy", will be a dictionary of
            astropy quantities. If "numpy", will be a dictionary of numpy arrays. Note that
            this method does not support all the formats available in :py:meth:`get_data <opencosmo.Dataset.get_data>`
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
        if datasets is None:
            datasets = list(self.keys())
        elif isinstance(datasets, str):
            datasets = [datasets]
        else:
            datasets = list(datasets)

        results = self.__map(
            "evaluate",
            func,
            vectorize=vectorize,
            insert=insert,
            format=format,
            construct=insert,
            **evaluate_kwargs,
        )
        if next(iter(results.values())) is None:
            return
        return results

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
        return self.__map("sort_by", column=column, invert=invert)

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns: u.Unit,
    ) -> Self:
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
        return self.__map("with_units", convention, conversions=conversions, **columns)
