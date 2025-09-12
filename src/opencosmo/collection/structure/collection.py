from __future__ import annotations

from typing import Any, Callable, Generator, Iterable, Mapping, Optional
from warnings import warn

import astropy  # type: ignore
import numpy as np

import opencosmo as oc
from opencosmo.collection.structure import evaluate
from opencosmo.collection.structure import io as sio
from opencosmo.dataset.column import DerivedColumn
from opencosmo.index import DataIndex, SimpleIndex
from opencosmo.io import io
from opencosmo.io.schemas import StructCollectionSchema
from opencosmo.parameters import HaccSimulationParameters
from opencosmo.spatial.protocols import Region

from .handler import LinkedDatasetHandler


def filter_source_by_dataset(
    dataset: oc.Dataset,
    source: oc.Dataset,
    header: oc.header.OpenCosmoHeader,
    *masks,
) -> oc.Dataset:
    masked_dataset = dataset.filter(*masks)
    linked_column: str
    if header.file.data_type == "halo_properties":
        linked_column = "fof_halo_tag"
    elif header.file.data_type == "galaxy_properties":
        linked_column = "gal_tag"

    tags = masked_dataset.select(linked_column).data
    new_source = source.filter(oc.col(linked_column).isin(tags))
    return new_source


class StructureCollection:
    """
    A collection of datasets that contain both high-level properties
    and lower level information (such as particles) for structures
    in the simulation. Currently these structures include halos
    and galaxies.

    Every structure collection has a halo_properties or galaxy_properties dataset
    that contains the high-level measured attribute of the structures. Certain
    operations (e.g. :py:meth:`sort_by <opencosmo.StructureCollection.sort_by>`
    operate on this dataset.
    """

    def __init__(
        self,
        source: oc.Dataset,
        header: oc.header.OpenCosmoHeader,
        datasets: Mapping[str, oc.Dataset | StructureCollection],
        links: dict[str, LinkedDatasetHandler],
        hide_source: bool = False,
        **kwargs,
    ):
        """
        Initialize a linked collection with the provided datasets and links.
        """

        self.__source = source
        self.__header = header
        self.__datasets = dict(datasets)
        self.__links = links
        self.__index = self.__source.index
        self.__hide_source = hide_source

        if isinstance(self.__datasets.get("galaxy_properties"), StructureCollection):
            self.__datasets["galaxies"] = self.__datasets.pop("galaxy_properties")
            self.__links["galaxies"] = self.__links.pop("galaxy_properties")

    def __repr__(self):
        structure_type = self.__header.file.data_type.split("_")[0] + "s"
        keys = list(self.keys())
        if len(keys) == 2:
            dtype_str = " and ".join(keys)
        else:
            dtype_str = ", ".join(keys[:-1]) + ", and " + keys[-1]
        return f"Collection of {structure_type} with {dtype_str}"

    def __len__(self):
        return len(self.__source)

    @classmethod
    def open(
        cls, targets: list[io.OpenTarget], ignore_empty=True, **kwargs
    ) -> StructureCollection:
        return sio.build_structure_collection(targets, ignore_empty)

    @classmethod
    def read(cls, *args, **kwargs) -> StructureCollection:
        raise NotImplementedError

    @property
    def header(self):
        return self.__header

    @property
    def dtype(self):
        structure_type = self.__header.file.data_type.split("_")[0]
        return structure_type

    @property
    def cosmology(self) -> astropy.cosmology.Cosmology:
        """
        The cosmology of the structure collection
        """
        return self.__source.cosmology

    @property
    def properties(self) -> list[str]:
        """
        The high-level properties that are available as part of the
        halo_properties or galaxy_properties dataset.
        """
        return self.__source.columns

    @property
    def redshift(self) -> float | tuple[float, float]:
        """
        For snapshots, return the redshift or redshift range
        this dataset was drawn from.

        Returns
        -------
        redshift: float | tuple[float, float]

        """
        return self.__header.file.redshift

    @property
    def simulation(self) -> HaccSimulationParameters:
        """
        Get the parameters of the simulation this dataset is drawn
        from.

        Returns
        -------
        parameters: opencosmo.parameters.HaccSimulationParameters
        """
        return self.__header.simulation

    def keys(self) -> list[str]:
        """
        Return the names of the datasets in this collection.
        """
        keys = list(self.__datasets.keys())
        if not self.__hide_source:
            keys.append(self.__source.dtype)
        return keys

    def values(self) -> list[oc.Dataset | StructureCollection]:
        """
        Return the datasets in this collection.
        """
        return [self[name] for name in self.keys()]

    def items(self) -> Generator[tuple[str, oc.Dataset | StructureCollection]]:
        """
        Return the names and datasets as key-value pairs.
        """

        for k, v in zip(self.keys(), self.values()):
            yield k, v

    def __getitem__(self, key: str) -> oc.Dataset | oc.StructureCollection:
        """
        Return the linked dataset with the given key.
        """
        if key not in self.keys():
            raise KeyError(f"Dataset {key} not found in collection.")
        elif key == self.__header.file.data_type:
            return self.__source

        index = self.__links[key].make_index(self.__index)
        return self.__datasets[key].with_index(index)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for dataset in self.values():
            try:
                dataset.__exit__(*args)
            except AttributeError:
                continue

    @property
    def region(self):
        return self.__source.region

    def bound(
        self, region: Region, select_by: Optional[str] = None
    ) -> StructureCollection:
        """
        Restrict this collection to only contain structures in the specified region.
        Querying will be done based on the halo  or galaxy centers, meaning some
        particles may fall outside the given region.

        See :doc:`spatial_ref` for details of how to construct regions.

        Parameters
        ----------
        region: opencosmo.spatial.Region

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

        bounded = self.__source.bound(region, select_by)
        return StructureCollection(
            bounded, self.__header, self.__datasets, self.__links, self.__hide_source
        )

    def evaluate(
        self,
        func: Callable,
        dataset: Optional[str] = None,
        format: str = "astropy",
        vectorize: bool = False,
        insert: bool = True,
        **evaluate_kwargs: Any,
    ):
        """
        Iterate over the structures in this collection and apply func to each,
        collecting the results into a new column. These values will be computed
        immediately rather than lazily. If your new column can be created from a
        simple algebraic combination of existing columns, use
        :py:meth:`with_new_columns <opencosmo.StructureCollection.with_new_columns>`.

        You can substantially improve the performance of this method by specifying
        which data is actually needed to do the computation. This method will
        automatically select the requested data, avoiding reading unneeded data
        from disk.

        The function passed to this method must take arguments that match the names
        of datasets that are stored in this collection. You can specify specific
        columns that are needed with keyword arguments to this function. For example:

        .. code-block:: python

            import opencosmo as oc
            import numpy as np
            collection = oc.open("haloproperties.hdf5", "haloparticles.hdf5")

            def computation(halo_properties, dm_particles):
                dx = np.mean(dm_particles.data["x"]) - halo_properties["fof_halo_center_x"]
                dy = np.mean(dm_particles.data["y"]) - halo_properties["fof_halo_center_y"]
                dz = np.mean(dm_particles.data["z"]) - halo_properties["fof_halo_center_z"]
                offset = np.sqrt(dx**2 + dy**2 + dz**2)
                return offset / halo_properties["sod_halo_radius"]

            collection = collection.evaluate(
                computation,
                name="offset",
                halo_properties=[
                    "fof_halo_center_x",
                    "fof_halo_center_y",
                    "fof_halo_center_z"
                    "sod_halo_radius"
                ],
                dm_particles=["x", "y", "z"]
            )

        The collection will now contain a column named "offset" with the results of the
        computation applied to each halo in the collection. Columns produced in this
        way will not respond to changes in unit convention.

        It is not required to pass a list of column names for a given dataset. If a list
        is not provided, all columns will be passed to the computation function.

        For more details and advanced usage see :ref:`Evaluating on Structure Collections`

        Parameters
        ----------

        func: Callable
            The function to evaluate on the rows in the dataset.

        dataset: Optional[str], default = None
            The dataset inside this collection to evaluate the function on. If none, assumes the function requires data from
            multiple datasets.

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False) if evaluating on aa single dataset.
            Has no effect if evaluating over structures, since structures require input from multiple datasets which will not in
            general be the same length.

        insert: bool, default = True
            If true, the data will be inserted as a column in the specified dataset, or the main "properties" dataset
            if no dataset is specified. The new column will have the same name as the function. Otherwise the data
            will be returned directly.

        format: str, default = astropy
            Whether to provide data to your function as "astropy" quantities or "numpy" arrays/scalars. Default "astropy"

        **evaluate_kwargs: any,
            Any additional arguments that are required for your function to run. These will be passed directly
            to the function as keyword arguments. If a kwarg is an array of values with the same length as the dataset,
            it will be treated as an additional column.

        """
        if dataset is not None:
            datasets = dataset.split(".", 1)
            ds = self[datasets[0]]
            if isinstance(ds, oc.Dataset) and len(datasets) > 1:
                raise ValueError("Datasets cannot be nested!")
            elif isinstance(ds, oc.Dataset):
                result = ds.evaluate(
                    func,
                    format=format,
                    vectorize=vectorize,
                    insert=insert,
                    **evaluate_kwargs,
                )
            elif isinstance(ds, StructureCollection):
                ds_name = datasets[1] if len(datasets) > 1 else None
                result = ds.evaluate(
                    func,
                    ds_name,
                    format=format,
                    vectorize=vectorize,
                    insert=insert,
                    **evaluate_kwargs,
                )

            if result is None or not insert:
                return result

            assert isinstance(result, (oc.Dataset, StructureCollection))
            if ds.dtype == self.__source.dtype:
                new_source = result
                new_datasets = self.__datasets
            else:
                new_source = self.__source
                new_datasets = {**self.__datasets, datasets[0]: result}
            return StructureCollection(
                new_source,
                self.__header,
                new_datasets,
                self.__links,
                self.__hide_source,
            )
        else:
            known_datasets = set(self.keys())
            kwarg_names = set(evaluate_kwargs.keys())

            requested_datasets = kwarg_names.intersection(known_datasets)
            other_kwarg_names = kwarg_names.difference(known_datasets)

            columns = {key: evaluate_kwargs[key] for key in requested_datasets}
            kwargs = {key: evaluate_kwargs[key] for key in other_kwarg_names}

            output = evaluate.visit_structure_collection(
                func, columns, self, format=format, evaluator_kwargs=kwargs
            )
            if not insert or output is None:
                return output
            return self.with_new_columns(**output, dataset=self.__source.dtype)

    def filter(self, *masks, on_galaxies: bool = False) -> StructureCollection:
        """
        Apply a filter to the halo or galaxy properties. Filters are constructed with
        :py:func:`opencosmo.col` and behave exactly as they would in
        :py:meth:`opencosmo.Dataset.filter`.

        If the collection contains both halos and galaxies, the filter can be applied to
        the galaxy properties dataset by setting `on_galaxies=True`. However this will
        filter for *halos* that host galaxies that match this filter. As a result,
        galxies that do not match this filter will remain if another galaxy in their
        host halo does match.

        See :ref:`Querying in Collections` for some examples.


        Parameters
        ----------
        *filters: Mask
            The filters to apply to the properties dataset constructed with
            :func:`opencosmo.col`.

        on_galaxies: bool, optional
            If True, the filter is applied to the galaxy properties dataset.

        Returns
        -------
        StructureCollection
            A new collection filtered by the given masks.

        Raises
        -------
        ValueError
            If on_galaxies is True but the collection does not contain
            a galaxy properties dataset.
        """
        if not masks:
            return self
        if not on_galaxies or self.__source.dtype == "galaxy_properties":
            filtered = self.__source.filter(*masks)
        elif "galaxy_properties" not in self.__datasets:
            raise ValueError("Dataset galaxy_properties not found in collection.")
        else:
            galaxy_properties = self["galaxy_properties"]
            assert isinstance(galaxy_properties, oc.Dataset)
            filtered = filter_source_by_dataset(
                galaxy_properties, self.__source, self.__header, *masks
            )
        return StructureCollection(
            filtered, self.__header, self.__datasets, self.__links, self.__hide_source
        )

    def select(
        self,
        columns: str | Iterable[str],
        dataset: str,
    ) -> StructureCollection:
        """
        Update a dataset in the collection collection to only include the
        columns specified.


        Parameters
        ----------
        columns : str | Iterable[str]
            The columns to select from the dataset.

        dataset : str
            The dataset to select from.

        Returns
        -------
        StructureCollection
            A new collection with only the selected columns for the specified dataset.

        Raises
        -------
        ValueError
            If the specified dataset is not found in the collection.
        """
        if dataset == self.__header.file.data_type:
            new_source = self.__source.select(columns)
            return StructureCollection(
                new_source, self.__header, self.__datasets, self.__links
            )

        elif dataset not in self.__datasets:
            raise ValueError(f"Dataset {dataset} not found in collection.")
        output_ds = self.__datasets[dataset]
        if not isinstance(output_ds, oc.Dataset):
            raise NotImplementedError

        new_dataset = output_ds.select(columns)
        return StructureCollection(
            self.__source,
            self.__header,
            {**self.__datasets, dataset: new_dataset},
            self.__links,
            self.__hide_source,
        )

    def drop(self, columns: str | Iterable[str], dataset: Optional[str] = None):
        """
        Update the linked collection by dropping the specified columns
        in the given dataset. If no dataset is specified, the properties dataset
        is used. For example, if this collection contains galaxies,
        calling this function without a "dataset" argument will select columns
        from the galaxy_properties dataset.


        Parameters
        ----------
        columns : str | Iterable[str]
            The columns to select from the dataset.

        dataset : str, optional
            The dataset to select from. If None, the properties dataset is used.

        Returns
        -------
        StructureCollection
            A new collection with only the selected columns for the specified dataset.

        Raises
        -------
        ValueError
            If the specified dataset is not found in the collection.
        """

        if dataset is None or dataset == self.__header.file.data_type:
            new_source = self.__source.drop(columns)
            return StructureCollection(
                new_source, self.__header, self.__datasets, self.__links
            )

        elif dataset not in self.__datasets:
            raise ValueError(f"Dataset {dataset} not found in collection.")
        output_ds = self.__datasets[dataset]
        new_dataset = output_ds.drop(columns)
        return StructureCollection(
            self.__source,
            self.__header,
            {**self.__datasets, dataset: new_dataset},
            self.__links,
            self.__hide_source,
        )

    def sort_by(self, column: str, invert: bool = False) -> StructureCollection:
        """
        Re-order the collection based on one of the structure collection's properties. Each
        StructureCollection contains a halo_properties or galaxy_properties dataset that
        contains the high-level measured properties of the structures in this collection.
        This method always operates on that dataset.

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
        result : StructureCollection
            A new StructureCollection ordered by the given column.

        """

        new_source = self.__source.sort_by(column, invert=invert)
        return StructureCollection(
            new_source,
            self.__header,
            self.__datasets,
            self.__links,
            self.__hide_source,
        )

    def with_units(self, convention: str):
        """
        Apply the given unit convention to the collection.
        See :py:meth:`opencosmo.Dataset.with_units`

        Parameters
        ----------
        convention : str
            The unit convention to apply. One of "unitless", "scalefree",
            "comoving", or "physical".

        Returns
        -------
        StructureCollection
            A new collection with the unit convention applied.
        """
        new_source = self.__source.with_units(convention)
        new_datasets = {
            key: dataset.with_units(convention)
            for key, dataset in self.__datasets.items()
        }
        return StructureCollection(
            new_source, self.__header, new_datasets, self.__links, self.__hide_source
        )

    def take(self, n: int, at: str = "random"):
        """
        Take some number of structures from the collection.
        See :py:meth:`opencosmo.Dataset.take`.

        Parameters
        ----------
        n : int
            The number of structures to take from the collection.
        at : str, optional
            The method to use to take the structures. One of "random", "first",
            or "last". Default is "random".

        Returns
        -------
        StructureCollection
            A new collection with the structures taken from the original.
        """
        new_source = self.__source.take(n, at)
        return StructureCollection(
            new_source,
            self.__header,
            self.__datasets,
            self.__links,
            self.__hide_source,
        )

    def take_range(self, start: int, end: int):
        new_source = self.__source.take_range(start, end)
        return StructureCollection(
            new_source, self.__header, self.__datasets, self.__links, self.__hide_source
        )

    def with_new_columns(self, dataset: str, **new_columns: DerivedColumn):
        """
        Add new column(s) to one of the datasets in this collection. This behaves
        exactly like :py:meth:`oc.Dataset.with_new_columns`, except that you must
        specify which dataset the columns should refer too.

        .. code-block:: python

            pe = oc.col("phi") * oc.col("mass")
            collection = collection.with_new_columns("dm_particles", pe=pe)

        Structure collections can hold other structure collections. For example, a
        collection of Halos may hold a structure collection that contians the galaxies
        of those halos. To update datasets within these collections, use dot syntax
        to specify a path:

        .. code-block:: python

            pe = oc.col("phi") * oc.col("mass")
            collection = collection.with_new_columns("galaxies.star_particles", pe=pe)

        You can also pass numpy arrays or astropy quantities:

        .. code-block:: python

            random_value = np.random.randint(0, 90, size=len(collection))
            random_quantity = random_value*u.deg

            collection = collection.with_new_columns("halo_properties",
                random_quantity=random_quantity)

        See :ref:`Adding Custom Columns` for more examples.


        Parameters
        ----------
        dataset : str
            The name of the dataset to add columns to

        ** columns: opencosmo.DerivedColumn
            The new columns

        Returns
        -------
        new_collection : opencosmo.StructureCollection
            This collection with the additional columns added

        Raise
        -----
        ValueError
            If the dataset is not found in this collection
        """
        path = dataset.split(".")
        if len(path) > 1:
            collection_name = path[0]
            if collection_name not in self.keys():
                raise ValueError(f"No collection {collection_name} found!")
            new_collection = self.__datasets[collection_name]
            if not isinstance(new_collection, StructureCollection):
                raise ValueError(f"{collection_name} is not a collection!")
            new_collection = new_collection.with_new_columns(
                ".".join(path[1:]), **new_columns
            )
            return StructureCollection(
                self.__source,
                self.__header,
                {**self.__datasets, collection_name: new_collection},
                self.__links,
                self.__hide_source,
            )

        if dataset == self.__source.dtype:
            new_source = self.__source.with_new_columns(**new_columns)
            return StructureCollection(
                new_source, self.__header, self.__datasets, self.__links
            )
        elif dataset not in self.__datasets.keys():
            raise ValueError(f"Dataset {dataset} not found in this collection!")

        ds = self.__datasets[dataset]

        if not isinstance(ds, oc.Dataset):
            raise ValueError(f"{dataset} is not a dataset!")

        new_ds = ds.with_new_columns(**new_columns)
        return StructureCollection(
            self.__source,
            self.__header,
            {**self.__datasets, dataset: new_ds},
            self.__links,
            self.__hide_source,
        )

    def with_index(self, index: DataIndex):
        new_source = self.__source.with_index(index)
        return StructureCollection(
            new_source, self.__header, self.__datasets, self.__links, self.__hide_source
        )

    def objects(
        self, data_types: Optional[Iterable[str]] = None, ignore_empty=True
    ) -> Iterable[dict[str, Any]]:
        """
        Iterate over the objects in this collection as pairs of
        (properties, datasets). For example, a halo collection could yield
        the halo properties and datasets for each of the associated partcles.

        If you don't need all the datasets, you can specify a list of data types
        for example:

        .. code-block:: python

            for row, particles in
                collection.objects(data_types=["gas_particles", "star_particles"]):
                # do work

        At each iteration, "row" will be a dictionary of halo properties with associated
        units, and "particles" will be a dictionary of datasets with the same keys as
        the data types.
        """
        if data_types is None:
            data_types = self.__datasets.keys()

        data_types = list(data_types)
        if not all(dt in self.__datasets for dt in data_types):
            raise ValueError("Some data types are not linked in the collection.")

        if len(self) == 0:
            warn("Tried to iterate over a collection with no structures in it!")
            return

        for row in self.__source.rows(attach_index=True):
            row = dict(row)
            idx = row.pop("raw_index")
            input_index = SimpleIndex(np.atleast_1d(idx))
            output = {
                key: self.__datasets[key].with_index(
                    self.__links[key].make_index(input_index)
                )
                for key in data_types
            }
            if not self.__hide_source:
                output.update({self.__source.dtype: row})
            yield output

    def with_datasets(self, datasets: list[str]):
        """
        Create a new collection out of a subset of the datasets in this collection.
        It is also possible to do this when you iterate over the collection with
        :py:meth:`StructureCollection.objects <opencosmo.StructureCollection.objects>`,
        however doing it up front may be more desirable if you don't plan to use
        the dropped datasets at any point.
        """

        if not isinstance(datasets, list):
            raise ValueError("Expected a list with at least one entries")

        known_datasets = set(self.keys())
        requested_datasets = set(datasets)
        if not requested_datasets.issubset(known_datasets):
            raise ValueError(f"Unknown datasets {requested_datasets - known_datasets}")

        if self.__source.dtype not in requested_datasets:
            hide_source = True
        else:
            hide_source = False
            requested_datasets.remove(self.__source.dtype)

        new_datasets = {name: self[name] for name in requested_datasets}
        new_links = {name: self.__links[name] for name in requested_datasets}
        return StructureCollection(
            self.__source, self.__header, new_datasets, new_links, hide_source
        )

    def halos(self, *args, **kwargs):
        """
        Alias for "objects" in the case that this StructureCollection contains halos.
        """
        if self.__source.dtype == "halo_properties":
            yield from self.objects(*args, **kwargs)
        else:
            raise AttributeError("This collection does not contain halos!")

    def galaxies(self, *args, **kwargs):
        """
        Alias for "objects" in the case that this StructureCollection contains galaxies
        """
        if self.__source.dtype == "galaxy_properties":
            yield from self.objects(*args, **kwargs)
        else:
            raise AttributeError("This collection does not contain galaxies!")

    def make_schema(self) -> StructCollectionSchema:
        schema = StructCollectionSchema()
        source_name = self.__source.dtype

        for name, dataset in self.items():
            ds_schema = dataset.make_schema()
            if name == "galaxies":
                name = "galaxy_properties"
            schema.add_child(ds_schema, name)

        for name, handler in self.__links.items():
            if name == "galaxies":
                name = "galaxy_properties"
            link_schema = handler.make_schema(name, self.__index)
            schema.insert(link_schema, f"{source_name}.{name}")

        return schema
