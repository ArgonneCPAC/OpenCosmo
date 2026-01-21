from __future__ import annotations

from collections import defaultdict
from functools import partial, reduce
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Mapping, Optional
from warnings import warn

import numpy as np

import opencosmo as oc
from opencosmo.collection.structure import evaluate
from opencosmo.collection.structure import io as sio
from opencosmo.index.unary import get_length
from opencosmo.io.schema import FileEntry, make_schema

from .handler import LinkHandler

if TYPE_CHECKING:
    import astropy
    import astropy.units as u

    from opencosmo.column.column import DerivedColumn
    from opencosmo.index import DataIndex
    from opencosmo.io import io
    from opencosmo.io.schema import Schema
    from opencosmo.mpi import MPI
    from opencosmo.parameters import HaccSimulationParameters
    from opencosmo.spatial.protocols import Region


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


def do_idx_update(data: np.ndarray, comm: Optional[MPI.Comm] = None):
    if comm is None:
        return np.arange(len(data))
    lengths = comm.allgather(len(data))
    offsets = np.insert(np.cumsum(lengths), 0, 0)
    offset = offsets[comm.Get_rank()]
    result = np.arange(offset, offset + len(data))
    return result


def do_start_update(data: np.ndarray, size: np.ndarray, comm: Optional[MPI.Comm]):
    psum = np.insert(np.cumsum(size), 0, 0)[:-1]
    if comm is None:
        return psum
    lengths = comm.allgather(np.sum(size))
    offsets = np.insert(np.cumsum(lengths), 0, 0)
    offset = offsets[comm.Get_rank()]
    return psum + offset


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
        hide_source: bool = False,
        link_handler: Optional[LinkHandler] = None,
        derived_columns: Optional[set[str]] = None,
        **kwargs,
    ):
        """
        Initialize a linked collection with the provided datasets and links.
        """

        self.__source = source
        self.__header = header
        self.__datasets = dict(datasets)
        self.__index = self.__source.index
        self.__hide_source = hide_source
        if isinstance(self.__datasets.get("galaxy_properties"), StructureCollection):
            self.__datasets["galaxies"] = self.__datasets.pop("galaxy_properties")

        if link_handler is None:
            self.__handler = LinkHandler.from_link_names(
                self.__source.meta_columns, "galaxies" in self.__datasets
            )
            datasets = self.__handler.prep_datasets(self.__source, self.__datasets)
        else:
            self.__handler = link_handler

        if derived_columns is None:
            derived_columns = set()
        self.__derived_columns = derived_columns

    def __get_datasets(self):
        """
        Methods should never access __datasets directly. Instead,
        get them through this method, which ensures all the rebuilding is
        done when necessary.
        """
        if self.__handler is None:
            return self.__datasets
        self.__datasets = self.__handler.rebuild_datasets(
            self.__source, self.__datasets
        )
        self.__handler = LinkHandler.from_link_names(
            self.__source.meta_columns, "galaxies" in self.__datasets
        )
        return self.__datasets

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
    def redshift(self) -> float | tuple[float, float] | None:
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
        if key == self.__header.file.data_type:
            return self.__source
        datasets = self.__get_datasets()
        if key not in datasets.keys():
            raise KeyError(f"Dataset {key} not found in collection.")
        return datasets[key]

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
        new_handler = self.__handler.make_derived(self.__source)
        return StructureCollection(
            bounded,
            self.__header,
            self.__datasets,
            self.__hide_source,
            new_handler,
            self.__derived_columns,
        )

    def evaluate(
        self,
        func: Callable,
        dataset: Optional[str] = None,
        format: str = "astropy",
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
        from disk. The semantics for specifying the columns is identical to
        :py:meth:`select <opencosmo.StructureCollection.select>`.

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
        computation applied to each halo in the collection.

        It is not required to pass a list of column names for a given dataset. If a list
        is not provided, all columns will be passed to the computation function. Data will
        be passed into the function as numpy arrays or astropy tables, depending on the
        value of the "format" argument. However if the evaluation involes a nested
        structure collection (e.g. a galaxy collection inside a structure collection)
        in addition to other datasets, the nested collection will be passed to your
        function as a StructureCollection.

        For more details and advanced usage see :ref:`Evaluating on Structure Collections`

        Parameters
        ----------

        func: Callable
            The function to evaluate on the rows in the dataset.

        dataset: Optional[str], default = None
            The dataset inside this collection to evaluate the function on. If none, assumes the function requires data from
            multiple datasets. You can visit a dataset inside a nested structure collection by passing the path
            separated by dots, for example "galaxies.star_particles". Data will be fed to the function on a structure-by-structure
            basis, and the output should be the same length as the input data.

        insert: bool, default = True
            If true, the data will be inserted as a column in the specified dataset. If no dataset is specified, insert
            into the "halo_properties" dataset if this collection contains halos, or the "galaxy properties" if this
            collection contains galaxies. If False, simply return the data.

        format: str, default = astropy
            Whether to provide data to your function as "astropy" quantities or "numpy" arrays/scalars. Default "astropy". Note that
            this method does not support all the formats available in :py:meth:`get_data <opencosmo.Dataset.get_data>`

        **evaluate_kwargs: any,
            Any additional arguments that are required for your function to run. These will be passed directly
            to the function as keyword arguments. If a kwarg is an array of values with the same length as the dataset,
            it will be treated as an additional column.

        """
        # Note: there are a few cases we support
        # 1. Evaluating on multiple datasets and inserting into halo_properties/galaxy_properties
        # 2. Evaluating on a single dataset and inserting into that dataset
        # 3. Evaluating on multiple dataset and inserting into one of them

        # The second of these can (and has) been made lazy. The 1st and 3rd are eager, for now.
        # If the user sets insert=False, everything is eager.
        if dataset is not None and dataset == self.__source.dtype:
            return self.evaluate_on_dataset(
                func, dataset=dataset, format=format, insert=insert, **evaluate_kwargs
            )

        if format not in ["astropy", "numpy"]:
            raise ValueError(f"Invalid format requested for data: {format}")

        if dataset is not None and dataset.startswith("galaxies"):
            # Nested structure collection, special case
            dataset_path = dataset.split(".")
            sub_dataset = None
            if len(dataset_path) == 2:
                sub_dataset = dataset_path[1]

            result = self[dataset_path[0]].evaluate(
                func, sub_dataset, format, insert, **evaluate_kwargs
            )
            if not insert:
                return result
            return StructureCollection(
                self.__source,
                self.__header,
                self.__get_datasets() | {dataset: result},
                self.__hide_source,
                self.__handler.make_derived(self.__source),
                self.__derived_columns,
            )

        parameter_names = set(signature(func).parameters.keys())
        required_datasets = parameter_names.intersection(self.keys())
        if not required_datasets and dataset is None:
            raise ValueError(
                "If your function does not take dataset names as arguments, you must specify which dataset you want to evaluate on!"
            )

        if dataset is not None and not required_datasets:
            # case two
            if dataset not in self.keys():
                raise ValueError(f"Unknown dataset {dataset}")
            ds = self[dataset]

            result = ds.evaluate(
                func,
                insert=insert,
                format=format,
                strategy="chunked",
                **evaluate_kwargs,
            )

            if not insert:
                return result
            assert isinstance(result, oc.Dataset)
            assert isinstance(ds, oc.Dataset)

            new_derived_columns = set(result.columns).difference(ds.columns)
            new_derived_columns_ = [f"{dataset}.{col}" for col in new_derived_columns]
            return StructureCollection(
                self.__source,
                self.__header,
                self.__get_datasets() | {dataset: result},
                self.__hide_source,
                self.__handler.make_derived(self.__source),
                self.__derived_columns.union(new_derived_columns_),
            )

        else:
            # case one/3

            output = evaluate.visit_structure_collection_eagerly(
                func,
                self,
                dataset=dataset,
                format=format,
                evaluate_kwargs=evaluate_kwargs,
                insert=insert,
            )
            if not insert or output is None:
                return output
            return self.with_new_columns(
                **output,
                dataset=dataset if dataset is not None else self.__source.dtype,
            )

    def evaluate_on_dataset(
        self,
        func: Callable,
        dataset: Optional[str] = None,
        vectorize: bool = False,
        format: str = "astropy",
        insert: bool = True,
        **evaluate_kwargs: Any,
    ):
        """
        Evaluate an expression on a specific dataset in this collection. This method is different from calling
        :py:meth:`evaulate <opencosmo.StructureCollection.evaluate>` with a :code:`dataset` argument
        in that this method does not apply the function on a per-structure basis. It is roughtly equivalent
        to the following code:

        .. code-block:: python

            results = collection[dataset_name].evaluate(func, format, vectorize, insert=False)
            collection = collection.with_new_columns(dataset_name, my_computed_value = results)

        Keep in mind that the following code:

        .. code-block:: python

            collection[dataset_name].evaluate(func, format, vectorize, insert=true)

        *does* produces a new dataset with the given new column, but this dataset will not be a part of the
        original collection.


        """

        ds: oc.Dataset | StructureCollection
        if dataset is None or dataset == self.__source.dtype:
            result = self.__source.evaluate(
                func, vectorize, insert, format, **evaluate_kwargs
            )
            if not insert:
                return result
            assert isinstance(result, oc.Dataset)
            return StructureCollection(
                result,
                self.__header,
                self.__datasets,
                self.__hide_source,
                self.__handler.make_derived(self.__source),
                self.__derived_columns,
            )

        ds_path = dataset.split(".")
        if ds_path[0] not in self.__datasets:
            raise ValueError(f"Unknown dataset {dataset}")
        ds = self.__datasets[ds_path[0]]
        if len(ds_path) > 1 and isinstance(ds, oc.Dataset):
            raise ValueError(
                f"Recieved {dataset} as the dataset argument but {ds_path[0]} is a dataset, not a collection!"
            )

        if len(ds_path) == 1 and isinstance(ds, oc.Dataset):
            result = ds.evaluate(func, vectorize, insert, format, **evaluate_kwargs)
            if not insert:
                return result
            assert isinstance(result, oc.Dataset)
            assert isinstance(ds, oc.Dataset)
            new_derived_columns = set(result.columns).difference(ds.columns)
            new_derived_columns_ = [f"{dataset}.{col}" for col in new_derived_columns]
            return StructureCollection(
                self.__source,
                self.__header,
                self.__datasets | {dataset: result},
                self.__hide_source,
                self.__handler.make_derived(self.__source),
                self.__derived_columns.union(new_derived_columns_),
            )
        elif len(ds_path) == 1 and isinstance(ds, oc.StructureCollection):
            result = ds.evaluate(func, None, format, insert)

        elif len(ds_path) > 1 and isinstance(ds, oc.StructureCollection):
            result = ds.evaluate_on_dataset(
                func, ".".join(dataset[1:]), vectorize, format, insert
            )

        if not insert:
            return result

        assert isinstance(result, (oc.Dataset, StructureCollection))
        return StructureCollection(
            self.__source,
            self.__header,
            self.__datasets | {ds_path[0]: result},
            self.__hide_source,
            self.__handler.make_derived(self.__source),
            self.__derived_columns,
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
            galaxy_properties = self["galaxy_properties"]
            assert isinstance(galaxy_properties, oc.Dataset)
            filtered = filter_source_by_dataset(
                galaxy_properties, self.__source, self.__header, *masks
            )

        new_handler = self.__handler.make_derived(self.__source)
        return StructureCollection(
            filtered,
            self.__header,
            self.__datasets,
            self.__hide_source,
            new_handler,
            self.__derived_columns,
        )

    def select(
        self, **column_selections: str | Iterable[str] | dict
    ) -> StructureCollection:
        """
        Update a dataset in the collection collection to only include the
        columns specified. The name of the arguments to this function should be
        dataset names. For example:

        .. code-block:: python

            collection = collection.select(
                halo_properties = ["fof_halo_mass", "sod_halo_mass", "sod_halo_cdelta"],
                dm_particles = ["x", "y", "z"]
            )

        Datasets that do not appear in the argument list will not be modified. You can
        remove entire datasets from the collection with
        :py:meth:`with_datasets <opencosmo.StructureCollection.with_datasets>`

        For nested structure collections, such as galaxies within halos, you can pass
        a nested dictionary:

        .. code-block:: python

            collection = oc.open("haloproperties.hdf5", "haloparticles.hdf5", "galaxyproperties.hdf5", "galaxyparticles.hdf5")

            collection = collection.select(
                halo_properties = ["fof_halo_mass", "sod_halo_mass", "sod_halo_cdelta"],
                dm_particles = ["x", "y", "z"]
                galaxies = {
                    "galaxy_properties": ["gal_mass_bar", "gal_mass_star"],
                    "star_particles": ["x", "y", "z"]
                }
            )


        Parameters
        ----------
        **column_selections : str | Iterable[str] | dict[str, Iterable[str]]
            The columns to select from a given dataset or sub-collection

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
        if not column_selections:
            return self
        new_source = self.__source
        new_datasets = {}
        for dataset, columns in column_selections.items():
            if dataset == self.__header.file.data_type:
                new_source = self.__source.select(columns)
                continue

            elif dataset not in self.__datasets:
                raise ValueError(f"Dataset {dataset} not found in collection.")

            new_ds = self.__datasets[dataset]

            if not isinstance(new_ds, oc.Dataset):
                if not isinstance(columns, dict):
                    raise ValueError(
                        "When working with nested structure collections, the argument should be a dictionary!"
                    )
                new_ds = new_ds.select(**columns)
            else:
                new_ds = new_ds.select(columns)

            new_datasets[dataset] = new_ds

        return StructureCollection(
            new_source,
            self.__header,
            self.__datasets | new_datasets,
            self.__hide_source,
            self.__handler.make_derived(self.__source),
            self.__derived_columns,
        )

    def drop(self, **columns_to_drop):
        """
        Update the linked collection by dropping the specified columns
        in the specified datasets. This method follows the exact same semantics as
        :py:meth:`StructureCollection.select <opencosmo.StructureCollection.select>`.
        Argument names should be datasets in this collection, and the argument
        values should be a string, list of strings, or dictionary.

        Datasets that are not included will not be modified. You can drop
        entire datasets with :py:meth:`with_datasets <opencosmo.StructureCollection.with_datasets>`

        Parameters
        ----------
        **columns_to_drop : str | Iterable[str]
            The columns to drop from the dataset.

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
        if not columns_to_drop:
            return self
        new_source = self.__source
        new_datasets = {}

        for dataset_name, columns in columns_to_drop.items():
            if dataset_name == self.__header.file.data_type:
                new_source = self.__source.drop(columns)
                continue

            elif dataset_name not in self.__datasets:
                raise ValueError(f"Dataset {dataset_name} not found in collection.")
            new_ds = self.__datasets[dataset_name]
            if isinstance(new_ds, oc.Dataset):
                new_ds = new_ds.drop(columns)
            elif isinstance(new_ds.StructureCollection):
                new_ds = new_ds.drop(**columns)

            new_datasets[dataset_name] = new_ds

        return StructureCollection(
            new_source,
            self.__header,
            self.__datasets | new_datasets,
            self.__hide_source,
            self.__handler.make_derived(self.__source),
            self.__derived_columns,
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
            self.__hide_source,
            self.__handler.make_derived(self.__source),
            self.__derived_columns,
        )

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **dataset_conversions: dict,
    ):
        """
        Apply the given unit convention to the collection, or convert a subset
        of the columns in one or more of these datasets into a compatible
        unit.

        Because this collection contains several datasets, you must specify
        the dataset when performing conversions. For example, the equivalent
        unit conversion to the final one in the example in
        :py:meth:`opencosmo.Dataset.with_units` looks like this:

        .. code-block:: python

            import astropy.units as u

            structures = structures.with_units(
                "physical",
                halo_properties={"fof_halo_mass": u.kg, "fof_halo_center_x": u.ly}
            )

        You can use :code:`conversions` to specify a conversion that applies to all
        columns in the collection with the given unit, or specify per-dataset conversions.
        Per-dataset conversions always take precedent over collection-wide conversions.
        For example:

        .. code-block:: python

            import astropy.units as u

            conversions = {u.Mpc: u.lyr}
            structures = structures.with_units(
                conversions=conversions
                halo_properties = {
                    "conversions": {u.Mpc: u.km},
                    "fof_halo_center_x": u.m
                }
            )

        In this example, all values in Mpc will be converted to lightyears, except in the "halo_properties" dataset,
        where they will be converted to kilometers. The column "fof_halo_center_x" in "halo_properties" will
        be converted to meters instead.

        For more information, see :doc:`units`

        Parameters
        ----------
        convention : str
            The unit convention to apply. One of "unitless", "scalefree",
            "comoving", or "physical".

        conversions : dict[astropy.units.Unit, astropy.units.Unit]
            Unit conversions to apply across all columns in the collection

        **dataset_conversion : dict
            Unit conversions apply to specific datasets in the collection.

        Returns
        -------
        StructureCollection
            A new collection with the unit convention applied.
        """
        if conversions:
            for ds_name in self.keys():
                ds_conversions = dataset_conversions.get(ds_name, {})
                new_ds_conversions = conversions | ds_conversions.get("conversions", {})
                ds_conversions["conversions"] = new_ds_conversions
                dataset_conversions[ds_name] = ds_conversions

        conversion_keys = set(dataset_conversions.keys())
        unknown = conversion_keys.difference(self.keys())
        if unknown:
            raise ValueError(f"Unknown datasets in conversions: {unknown}")

        if self.__source.dtype in conversion_keys or (
            not conversion_keys and convention is not None
        ):
            new_source = self.__source.with_units(
                convention, **dataset_conversions.get(self.__source.dtype, {})
            )
        else:
            new_source = self.__source
        new_datasets = {}
        for key, dataset in self.__datasets.items():
            ds_conversions = dataset_conversions.get(key, {})
            if convention is None and not ds_conversions:
                new_datasets[key] = dataset.with_units()
                continue
            new_ds = dataset.with_units(convention, **ds_conversions)
            new_datasets[key] = new_ds

        return StructureCollection(
            new_source,
            self.__header,
            new_datasets,
            self.__hide_source,
            self.__handler.make_derived(self.__source),
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
        new_handler = self.__handler.make_derived(self.__source)

        return StructureCollection(
            new_source,
            self.__header,
            self.__datasets,
            self.__hide_source,
            new_handler,
            self.__derived_columns,
        )

    def take_range(self, start: int, end: int):
        """
        Create a new collection from a row range in this collection. We use standard
        indexing conventions, so the rows included will be start -> end - 1.

        Parameters
        ----------
        start : int
            The first row to get.
        end : int
            The last row to get.

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
        new_source = self.__source.take_range(start, end)
        return StructureCollection(
            new_source,
            self.__header,
            self.__datasets,
            self.__hide_source,
            self.__handler.make_derived(self.__source),
            self.__derived_columns,
        )

    def take_rows(self, rows: np.ndarray | DataIndex):
        """
        Take the rows of this collection  specified by the :code:`rows` argument.
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
        new_source = self.__source.take_rows(rows)
        return StructureCollection(
            new_source,
            self.__header,
            self.__datasets,
            self.__hide_source,
            self.__handler.make_derived(self.__source),
            self.__derived_columns,
        )

    def with_new_columns(
        self,
        dataset: str,
        descriptions: str | dict[str, str] = {},
        **new_columns: DerivedColumn,
    ):
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

        descriptions : str | dict[str, str], optional
            Descriptions for the new columns. These descriptions will be accessible through
            :py:attr:`Dataset.descriptions <opencosmo.Dataset.descriptions>`. If a dictionary,
            should have keys matching the column names.

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
                ".".join(path[1:]), descriptions=descriptions, **new_columns
            )
            return StructureCollection(
                self.__source,
                self.__header,
                {**self.__datasets, collection_name: new_collection},
                self.__hide_source,
                self.__handler.make_derived(self.__source),
            )

        if dataset == self.__source.dtype:
            new_source = self.__source.with_new_columns(
                **new_columns, descriptions=descriptions
            )
            return StructureCollection(
                new_source,
                self.__header,
                self.__datasets,
                self.__hide_source,
                self.__handler.make_derived(self.__source),
                self.__derived_columns,
            )
        elif dataset not in self.__datasets.keys():
            raise ValueError(f"Dataset {dataset} not found in this collection!")

        new_im_cols = {
            name for name, col in new_columns.items() if isinstance(col, np.ndarray)
        }
        ds = self.__datasets[dataset]

        if not isinstance(ds, oc.Dataset):
            raise ValueError(f"{dataset} is not a dataset!")

        new_ds = ds.with_new_columns(**new_columns, descriptions=descriptions)
        new_derived_columns = (
            set(new_ds.columns).difference(ds.columns).difference(new_im_cols)
        )
        new_derived_columns_ = [f"{dataset}.{col}" for col in new_derived_columns]

        return StructureCollection(
            self.__source,
            self.__header,
            {**self.__datasets, dataset: new_ds},
            self.__hide_source,
            self.__handler.make_derived(self.__source),
            self.__derived_columns.union(new_derived_columns_),
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

            for halo in collection.objects(data_types=["halo_properties", "gas_particles", "star_particles"]):
                # do work

        At each iteration, :code:`halo` will be a dictionary with halo properties, gas_particles,
        and star particles. The "halo_properties" entry will itself be a dictionary with the halo's properties,
        while "gas_particles" and "star_particles" will be full :py:meth:`Datasets <opencosmo.Dataset>`.
        """
        if data_types is None:
            data_types = self.__datasets.keys()

        data_types = list(data_types)
        if not all(dt in self.__datasets for dt in data_types):
            raise ValueError("Some data types are not linked in the collection.")

        if len(self) == 0:
            warn("Tried to iterate over a collection with no structures in it!")
            return

        metadata_columns: list[str] = reduce(
            lambda acc, key: acc + self.__handler.columns[key], data_types, []
        )
        datasets = self.__get_datasets()
        rs = {name: 0 for name in self.__datasets.keys()}

        columns_to_collect: dict[str, dict[str, list[np.ndarray]]] = defaultdict(dict)
        for column in self.__derived_columns:
            name_parts = column.split(".")
            columns_to_collect[name_parts[0]][name_parts[1]] = []

        try:
            for row in self.__source.rows(metadata_columns=metadata_columns):
                row = dict(row)
                links = self.__handler.parse(row)
                output = {}
                for name, index in links.items():
                    ilength = get_length(index)
                    output[name] = datasets[name].take_range(
                        rs[name], rs[name] + ilength
                    )
                    rs[name] += ilength

                if not self.__hide_source:
                    output.update({self.__source.dtype: row})
                if not output:
                    continue

                for ds_name, ds_columns_to_collect in columns_to_collect.items():
                    if ds_name not in output:
                        continue
                    column_names = set(output[ds_name].columns).intersection(
                        ds_columns_to_collect.keys()
                    )
                    data = output[ds_name].select(column_names).get_data()
                    if not isinstance(data, dict):
                        ds_columns_to_collect[
                            next(iter(ds_columns_to_collect.keys()))
                        ].append(data)
                        continue

                    for colname, coldata in data.items():
                        ds_columns_to_collect[colname].append(coldata)

                yield output

            new_datasets = self.__get_datasets()
            for ds_name, collected_data in columns_to_collect.items():
                ds_length = len(new_datasets[ds_name])
                ds_data = {
                    name: np.concatenate(cd)
                    for name, cd in collected_data.items()
                    if len(cd) > 0
                }
                ds_data = {
                    name: d.reshape((ds_length, -1)) if len(d) > ds_length else d
                    for name, d in ds_data.items()
                }
                if not ds_data:
                    continue
                self.__derived_columns = self.__derived_columns.difference(
                    f"{ds_name}.{name}" for name in ds_data.keys()
                )
                descriptions = {
                    name: new_datasets[ds_name].descriptions[name]
                    for name in ds_data.keys()
                }

                new_dataset = (
                    new_datasets[ds_name]
                    .drop(ds_data.keys())
                    .with_new_columns(descriptions=descriptions, **ds_data)
                )
                new_datasets[ds_name] = new_dataset
            self.__datasets = new_datasets
        except GeneratorExit:
            pass
        except BaseException:
            raise

    def with_datasets(self, datasets: list[str]):
        """
        Create a new collection out of a subset of the datasets in this collection.
        It is also possible to do this when you iterate over the collection with
        :py:meth:`StructureCollection.objects <opencosmo.StructureCollection.objects>`,
        however doing it up front may be more desirable if you don't plan to use
        the dropped datasets at any point.
        """

        if not isinstance(datasets, list):
            raise ValueError("Expected a list with at least one entry")

        known_datasets = set(self.keys())
        requested_datasets = set(datasets)
        if not requested_datasets.issubset(known_datasets):
            raise ValueError(f"Unknown datasets {requested_datasets - known_datasets}")

        if self.__source.dtype not in requested_datasets:
            hide_source = True
        else:
            hide_source = False
            requested_datasets.remove(self.__source.dtype)

        new_datasets = {name: self.__datasets[name] for name in requested_datasets}
        return StructureCollection(
            self.__source,
            self.__header,
            new_datasets,
            hide_source,
            self.__handler.make_derived(self.__source),
            self.__derived_columns,
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

    def make_schema(self, name: Optional[str] = None) -> Schema:
        children = {}
        source_name = self.__source.dtype
        datasets = self.__handler.resort(self.__source, self.__get_datasets())

        source_schema = self.__source.make_schema()
        for colname, column in source_schema.children["data_linked"].columns.items():
            if "idx" in colname:
                column.set_transformation(do_idx_update)
            elif "start" in colname:
                size_colname = colname.replace("start", "size")
                size_data = (
                    source_schema.children["data_linked"].columns[size_colname].data
                )
                updater = partial(do_start_update, size=size_data)
                column.set_transformation(updater)

        children[source_name] = source_schema

        for name, dataset in datasets.items():
            if name == "galaxies":
                name = "galaxy_properties"
            ds_schema = dataset.make_schema()
            if not isinstance(dataset, StructureCollection):
                children[name] = ds_schema
                continue
            for grandchild_name, grandchild in ds_schema.children.items():
                if "properties" in grandchild_name:
                    children[grandchild_name] = grandchild
                else:
                    children[f"{name}_{grandchild_name}"] = grandchild

        if name is None:
            name = ""
        return make_schema(name, FileEntry.STRUCTURE_COLLECTION, children=children)
