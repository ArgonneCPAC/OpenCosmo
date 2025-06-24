from __future__ import annotations

from typing import Any, Iterable, Optional

import astropy  # type: ignore
import h5py

import opencosmo as oc
from opencosmo import structure as s
from opencosmo.dataset.col import DerivedColumn
from opencosmo.io.schemas import StructCollectionSchema
from opencosmo.parameters import SimulationParameters
from opencosmo.spatial.protocols import Region


def filter_properties_by_dataset(
    dataset: oc.Dataset,
    properties: oc.Dataset,
    header: oc.header.OpenCosmoHeader,
    *masks,
) -> oc.Dataset:
    masked_dataset = dataset.filter(*masks)
    if header.file.data_type == "halo_properties":
        linked_column = "fof_halo_tag"
    elif header.file.data_type == "galaxy_properties":
        linked_column = "gal_tag"

    tags = masked_dataset.select(linked_column).data
    new_properties = properties.filter(oc.col(linked_column).isin(tags))
    return new_properties


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
        properties: oc.Dataset,
        header: oc.header.OpenCosmoHeader,
        handlers: dict[str, s.LinkedDatasetHandler],
        *args,
        **kwargs,
    ):
        """
        Initialize a linked collection with the provided datasets and links.
        """

        self.__properties = properties
        self.__header = header
        self.__handlers = handlers
        self.__index = self.__properties.index

    def __repr__(self):
        structure_type = self.__header.file.data_type.split("_")[0] + "s"
        dtype_str = ", ".join(self.__handlers.keys())
        return f"Collection of {structure_type} with linked datasets {dtype_str}"

    def __len__(self):
        return len(self.__properties)

    @classmethod
    def open(
        cls,
        file: h5py.File,
        datasets_to_get: Optional[Iterable[str]] = None,
    ) -> StructureCollection:
        return s.open_linked_file(file, datasets_to_get)

    @classmethod
    def read(cls, *args, **kwargs) -> StructureCollection:
        raise NotImplementedError

    @property
    def cosmology(self) -> astropy.cosmology.Cosmology:
        """
        The cosmology of the structure collection
        """
        return self.__properties.cosmology

    @property
    def redshift(self) -> float | tuple[float, float]:
        """
        For snapshots, return the redshift of the slice
        this dataset was drawn from. For lightcones, return the redshift
        range.

        Returns:
        --------
        redshift: float | tuple[float, float]

        """
        return self.__header.file.redshift

    @property
    def simulation(self) -> SimulationParameters:
        """
        Get the parameters of the simulation this dataset is drawn
        from.

        Returns
        -------
        parameters: opencosmo.parameters.SimulationParameters
        """
        return self.__header.simulation

    @property
    def properties(self) -> oc.Dataset:
        """
        The properties dataset of the collection. Either, halo properties
        or galaxy properties.
        """
        return self.__properties

    def keys(self) -> list[str]:
        """
        Return the keys of the linked datasets.
        """
        return list(self.__handlers.keys()) + [self.__properties.dtype]

    def values(self) -> list[oc.Dataset]:
        """
        Return the linked datasets.
        """
        return [self.__properties] + [
            handler.get_dataset(self.__index) for handler in self.__handlers.values()
        ]

    def items(self) -> list[tuple[str, oc.Dataset]]:
        """
        Return the linked datasets as key-value pairs.
        """
        return [
            (key, handler.get_dataset(self.__index))
            for key, handler in self.__handlers.items()
        ]

    def __getitem__(self, key: str) -> oc.Dataset:
        """
        Return the linked dataset with the given key.
        """
        if key == self.__header.file.data_type:
            return self.__properties
        elif key not in self.__handlers:
            raise KeyError(f"Dataset {key} not found in collection.")
        index = self.__properties.index
        return self.__handlers[key].get_dataset(index)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for dataset in self.values():
            try:
                dataset.__exit__(*args)
            except AttributeError:
                continue

    def bound(
        self, region: Region, select_by: Optional[str] = None
    ) -> StructureCollection:
        """
        Restrict this collection to only contain structures in the specified region.
        Querying will be done based on the halo centers, meaning some particles may
        fall outside the given region.

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

        bounded = self.__properties.bound(region, select_by)
        return StructureCollection(bounded, self.__header, self.__handlers)

    def filter(self, *masks, on_galaxies: bool = False) -> StructureCollection:
        """
        Apply a filter to the properties dataset and propagate it to the linked
        datasets. Filters are constructed with :py:func:`opencosmo.col` and behave
        exactly as they would in `opencosmo.Dataset.filter`.

        If the collection contains both halos and galaxies, the filter can be applied to
        the galaxy properties dataset by setting `on_galaxies=True`. However this will
        filter for *halos* that host galaxies that match this filter. As a result,
        galxies that do not match this filter will remain if another galaxy in their
        host halo does match.


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
        if not on_galaxies or self.__properties.dtype == "galaxy_properties":
            filtered = self.__properties.filter(*masks)
        elif "galaxy_properties" not in self.__handlers:
            raise ValueError("Dataset galaxy_properties not found in collection.")
        else:
            filtered = filter_properties_by_dataset(
                self["galaxy_properties"], self.__properties, self.__header, *masks
            )
        return StructureCollection(
            filtered,
            self.__header,
            self.__handlers,
        )

    def select(
        self, columns: str | Iterable[str], dataset: Optional[str] = None
    ) -> StructureCollection:
        """
        Update the linked collection to only include the columns specified
        in the given dataset. If no dataset is specified, the properties dataset
        is used.

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
            new_properties = self.__properties.select(columns)
            return StructureCollection(
                new_properties,
                self.__header,
                self.__handlers,
            )

        elif dataset not in self.__handlers:
            raise ValueError(f"Dataset {dataset} not found in collection.")
        handler = self.__handlers[dataset]
        new_handler = handler.select(columns)
        return StructureCollection(
            self.__properties, self.__header, {**self.__handlers, dataset: new_handler}
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
        new_properties = self.__properties.with_units(convention)
        new_handlers = {
            key: handler.with_units(convention)
            for key, handler in self.__handlers.items()
        }
        return StructureCollection(
            new_properties,
            self.__header,
            new_handlers,
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
        new_properties = self.__properties.take(n, at)
        return StructureCollection(
            new_properties,
            self.__header,
            self.__handlers,
        )

    def with_new_columns(self, dataset: str, **new_columns: DerivedColumn):
        """
        Add new column(s) to one of the datasets in this collection. This behaves
        exactly like :py:meth:`oc.Dataset.with_new_columns`, except that you must
        specify which dataset the columns should refer too.

        .. code-block:: python

            pe = oc.col("phi") * oc.col("mass")
            collection = collection.with_new_columns("dm_particles", pe=pe)

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
        if dataset == self.__properties.dtype:
            new_properties = self.__properties.with_new_columns(**new_columns)
            return StructureCollection(new_properties, self.__header, self.__handlers)
        elif dataset not in self.__handlers.keys():
            raise ValueError(f"Dataset {dataset} not found in this collection!")

        new_handlers = {
            dataset: self.__handlers[dataset].with_new_columns(**new_columns)
        }
        for key, handler in self.__handlers.items():
            if key == dataset:
                continue
            new_handlers[key] = handler
        return StructureCollection(self.__properties, self.__header, new_handlers)

    def objects(
        self,
        data_types: Optional[Iterable[str]] = None,
    ) -> Iterable[tuple[dict[str, Any], oc.Dataset | dict[str, oc.Dataset]]]:
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
            handlers = self.__handlers
        elif not all(dt in self.__handlers for dt in data_types):
            raise ValueError("Some data types are not linked in the collection.")
        else:
            handlers = {dt: self.__handlers[dt] for dt in data_types}

        for i, row in enumerate(self.__properties.rows()):
            index = self.__properties.index[i]
            output = {
                key: handler.get_dataset(index) for key, handler in handlers.items()
            }
            if not any(len(v) for v in output.values()):
                continue
            if len(output) == 1:
                yield row, next(iter(output.values()))
            else:
                yield row, output

    def make_schema(self) -> StructCollectionSchema:
        schema = StructCollectionSchema(self.__header)
        properties_name = self.properties.dtype
        for name, dataset in self.items():
            ds_schema = dataset.make_schema()
            schema.add_child(ds_schema, name)

        properties_schema = self.properties.make_schema()
        schema.add_child(properties_schema, properties_name)

        for name, handler in self.__handlers.items():
            link_schema = handler.make_schema(name, self.__index)
            schema.insert(link_schema, f"{properties_name}.{name}")

        return schema
