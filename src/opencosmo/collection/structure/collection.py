from __future__ import annotations

from typing import Any, Generator, Iterable, Mapping, Optional
from warnings import warn

import astropy  # type: ignore

import opencosmo as oc
from opencosmo.collection.structure import io as sio
from opencosmo.dataset.col import DerivedColumn
from opencosmo.index import DataIndex
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

    For now, these are always a combination of a properties dataset
    and several particle or profile datasets.
    """

    def __init__(
        self,
        source: oc.Dataset,
        header: oc.header.OpenCosmoHeader,
        datasets: Mapping[str, oc.Dataset | StructureCollection],
        links: dict[str, LinkedDatasetHandler],
        *args,
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

        if isinstance(self.__datasets.get("galaxy_properties"), StructureCollection):
            self.__datasets["galaxies"] = self.__datasets.pop("galaxy_properties")
            self.__links["galaxies"] = self.__links.pop("galaxy_properties")

    def __repr__(self):
        structure_type = self.__header.file.data_type.split("_")[0] + "s"
        keys = list(self.keys())
        if len(keys) == 2:
            dtype_str = " and ".join(keys)
        else:
            dtype_str = ", ".join(keys[:-1]) + ", and" + keys[-1]
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
        raise NotImplementedError()

    @property
    def cosmology(self) -> astropy.cosmology.Cosmology:
        """
        The cosmology of the structure collection
        """
        return self.__source.cosmology

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
        return [self.__source.dtype] + list(self.__datasets.keys())

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

    def __getitem__(self, key: str) -> oc.Dataset:
        """
        Return the linked dataset with the given key.
        """
        if key == self.__header.file.data_type:
            return self.__source
        elif key not in self.__datasets:
            raise KeyError(f"Dataset {key} not found in collection.")

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
            bounded, self.__header, self.__datasets, self.__links
        )

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
            filtered = filter_source_by_dataset(
                self["galaxy_properties"], self.__source, self.__header, *masks
            )
        return StructureCollection(
            filtered, self.__header, self.__datasets, self.__links
        )

    def select(
        self, columns: str | Iterable[str], dataset: Optional[str] = None
    ) -> StructureCollection:
        """
        Update the linked collection to only include the columns specified
        in the given dataset. If no dataset is specified the properties of the
        structure will be used. For example, if this collection contains halos,
        calling this function without a "dataset" argument will select columns
        from the halo_properties dataset.

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
            new_source = self.__source.select(columns)
            return StructureCollection(
                new_source, self.__header, self.__datasets, self.__links
            )

        elif dataset not in self.__datasets:
            raise ValueError(f"Dataset {dataset} not found in collection.")
        output_ds = self.__datasets[dataset]
        new_dataset = output_ds.select(columns)
        return StructureCollection(
            self.__source,
            self.__header,
            {**self.__datasets, dataset: new_dataset},
            self.__links,
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
            new_source, self.__header, new_datasets, self.__links
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
        )

    def take_range(self, start: int, end: int):
        new_source = self.__source.take_range(start, end)
        return StructureCollection(
            new_source, self.__header, self.__datasets, self.__links
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

        See :ref:`Creating New Columns in Collections` for examples.

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
        )

    def with_index(self, index: DataIndex):
        new_source = self.__source.with_index(index)
        return StructureCollection(
            new_source, self.__header, self.__datasets, self.__links
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

        for i, row in enumerate(self.__source.rows()):
            index = self.__source.index[i]
            output = {
                key: self.__datasets[key].with_index(
                    self.__links[key].make_index(index)
                )
                for key in data_types
            }
            output.update({self.__source.dtype: row})
            yield output

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
        schema = StructCollectionSchema(self.__header)
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
