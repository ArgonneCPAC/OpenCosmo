from __future__ import annotations

from typing import Iterable, Mapping, Optional

import h5py
from astropy.cosmology import Cosmology  # type: ignore

import opencosmo as oc
from opencosmo.collection.protocols import Collection
from opencosmo.dataset.col import Mask
from opencosmo.dataset.handler import DatasetHandler
from opencosmo.dataset.state import DatasetState
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.index import ChunkedIndex
from opencosmo.io.protocols import DataSchema
from opencosmo.io.schemas import SimCollectionSchema
from opencosmo.parameters import SimulationParameters
from opencosmo.spatial.protocols import Region
from opencosmo.spatial.tree import open_tree
from opencosmo.structure import StructureCollection
from opencosmo.transformations import units as u


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

    def __init__(self, datasets: Mapping[str, oc.Dataset | Collection]):
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
            if isinstance(v, (SimulationCollection, StructureCollection))
        )
        n_datasets = sum(1 for v in self.values() if isinstance(v, oc.Dataset))
        return (
            f"SimulationCollection({n_collections} collections, {n_datasets} datasets)"
        )

    @classmethod
    def open(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> SimulationCollection:
        if datasets_to_get is not None:
            verify_datasets_exist(file, datasets_to_get)
            names = datasets_to_get
        else:
            names = list(filter(lambda x: x != "header", file.keys()))
        datasets = {name: oc.open(file[name]) for name in names}
        return cls(datasets)

    @classmethod
    def read(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> SimulationCollection:
        if datasets_to_get is not None:
            verify_datasets_exist(file, datasets_to_get)
            names = datasets_to_get
        else:
            names = list(filter(lambda x: x != "header", file.keys()))

        datasets = {name: read_single_dataset(file, name) for name in names}
        return cls(datasets)

    def make_schema(self) -> DataSchema:
        schema = SimCollectionSchema()
        for name, dataset in self.items():
            ds_schema = dataset.make_schema()
            schema.add_child(ds_schema, name)

        return schema

    def __map(self, method, *args, **kwargs):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        output = {k: getattr(v, method)(*args, **kwargs) for k, v in self.items()}
        return SimulationCollection(output)

    def __map_attribute(self, attribute):
        return {k: getattr(v, attribute) for k, v in self.items()}

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
    def simulation(self) -> dict[str, SimulationParameters]:
        """
        Get the simulation parameters for the simulations in the collection

        Returns
        --------
        simulation_parameters: dict[str, opencosmo.parameters.SimulationParameters]
        """

        return self.__map_attribute("simulation")

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
        dataset: opencosmo.Dataset
            The portion of the dataset inside the selected region

        """
        return self.__map("bound", region, select_by)

    def filter(self, *masks: Mask, **kwargs) -> SimulationCollection:
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

    def select(self, *args, **kwargs) -> SimulationCollection:
        """
        Select a subset of the datasets in the collection. This method
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

        """
        return self.__map("select", *args, **kwargs)

    def take(self, n: int, at: str = "random") -> SimulationCollection:
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

    def with_new_columns(self, *args, **kwargs):
        """
        Update the datasets within this collection with a set of new columns.
        This method simply calls :py:meth:`opencosmo.Dataset.with_new_columns` or
        :py:meth:`opencosmo.StructureCollection.with_new_columns`, as appropriate.
        """
        return self.__map("with_new_columns", *args, **kwargs)

    def with_units(self, convention: str) -> SimulationCollection:
        """
        Transform all datasets or collections to use the given unit convention. This
        method behaves exactly like :meth:`opencosmo.Dataset.with_units`.

        Parameters
        ----------
        convention: str
            The unit convention to use. One of "unitless",
            "scalefree", "comoving", or "physical".

        """
        return self.__map("with_units", convention)


def read_single_dataset(
    file: h5py.File, dataset_key: str, header: Optional[OpenCosmoHeader] = None
):
    """
    Read a single dataset from a multi-dataset file
    """
    if dataset_key not in file.keys():
        raise ValueError(f"No group named '{dataset_key}' found in file.")

    if header is None:
        header = read_header(file[dataset_key])

    try:
        tree = open_tree(file[dataset_key], header.simulation.box_size)
    except ValueError:
        tree = None
    p1 = (0, 0, 0)
    p2 = tuple(header.simulation.box_size for _ in range(3))
    sim_box = oc.make_box(p1, p2)
    im_file = h5py.File.in_memory()
    file.copy(dataset_key, im_file)
    handler = DatasetHandler(im_file, dataset_key)

    builders, base_unit_transformations = u.get_default_unit_transformations(
        file[dataset_key], header
    )
    index = ChunkedIndex.from_size(len(handler))
    state = DatasetState(
        base_unit_transformations, builders, index, u.UnitConvention.COMOVING, sim_box
    )

    return oc.Dataset(handler, header, state, tree)
