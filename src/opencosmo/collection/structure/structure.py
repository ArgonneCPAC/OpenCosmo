from __future__ import annotations

from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Mapping, Optional
from warnings import warn

import numpy as np

import opencosmo as oc
from opencosmo.collection.structure import evaluate
from opencosmo.collection.structure import io as sio
from opencosmo.index import ChunkedIndex, SimpleIndex
from opencosmo.io.schemas import StructCollectionSchema

if TYPE_CHECKING:
    import astropy
    import astropy.units as u

    from opencosmo.column.column import DerivedColumn
    from opencosmo.index import DataIndex
    from opencosmo.io import io
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


LINK_ALIASES = {  # Left: Name in file, right: Name in collection
    "sodbighaloparticles_star_particles": "star_particles",
    "sodbighaloparticles_dm_particles": "dm_particles",
    "sodbighaloparticles_gravity_particles": "gravity_particles",
    "sodbighaloparticles_agn_particles": "agn_particles",
    "sodbighaloparticles_gas_particles": "gas_particles",
    "sod_profile": "halo_profiles",
    "galaxyproperties": "galaxy_properties",
    "galaxyparticles_star_particles": "star_particles",
}


def create_start_size(data, start_name, size_name):
    start = data.pop(start_name, None)
    size = data.pop(size_name, None)
    if start is None:
        return None
    if isinstance(start, np.ndarray):
        return ChunkedIndex(start, size)
    if size == 0:
        return None
    return ChunkedIndex.single_chunk(start, size)


def create_idx(data, idx_name):
    idx = data.pop(idx_name, None)
    if idx is None:
        return None

    if isinstance(idx, np.ndarray):
        return SimpleIndex(idx)
    elif idx == -1:
        return None
    return SimpleIndex(np.atleast_1d(idx))


def make_links(keys, rename_galaxies=False):
    starts = list(filter(lambda key: "start" in key, keys))
    sizes = list(filter(lambda key: "size" in key, keys))
    idxs = list(filter(lambda key: "idx" in key, keys))

    starts = set(map(lambda key: key[:-6], starts))
    sizes = set(map(lambda key: key[:-5], sizes))
    idxs = set(map(lambda key: key[:-4], idxs))

    assert starts == sizes
    output = {}
    columns = {}
    for name in starts:
        output[LINK_ALIASES[name]] = partial(
            create_start_size, start_name=f"{name}_start", size_name=f"{name}_size"
        )
        columns[LINK_ALIASES[name]] = [f"{name}_start", f"{name}_size"]

    for name in idxs:
        output[LINK_ALIASES[name]] = partial(create_idx, idx_name=f"{name}_idx")
        columns[LINK_ALIASES[name]] = [f"{name}_idx"]

    if rename_galaxies and "galaxy_properties" in output:
        output["galaxies"] = output.pop("galaxy_properties")
        columns["galaxies"] = columns.pop("galaxy_properties")
    return output, columns


class LinkHandler:
    def __init__(self, links: Iterable[str], rename_galaxies=False):
        self.links, self.columns = make_links(links, rename_galaxies)

    def parse(self, data: dict[str, Any]):
        output = {}
        for name, handler in self.links.items():
            result = handler(data)
            if result is not None:
                output[name] = result
        return output


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

        self.__handler = LinkHandler(
            self.__source.meta_columns, "galaxies" in self.__datasets
        )

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
        ds = self.__datasets[key]
        if isinstance(ds, StructureCollection):
            idx = ds.__source.index
            if not idx.range()[0] == 0 or len(ds) != idx.range()[1]:
                return ds

        meta = self.__source.get_metadata(self.__handler.columns[key])
        indices = self.__handler.parse(meta)
        ds = self.__datasets[key]
        index = indices[key]

        return self.__datasets[key].take_rows(index)

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
            bounded, self.__header, self.__datasets, self.__hide_source
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
        computation applied to each halo in the collection. Columns produced in this
        way will not respond to changes in unit convention.

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
            filtered, self.__header, self.__datasets, self.__hide_source
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
            not conversion_keys and convention is None
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
            new_source, self.__header, new_datasets, self.__hide_source
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
            self.__hide_source,
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
            new_source, self.__header, self.__datasets, self.__hide_source
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
            new_source, self.__header, self.__datasets, self.__hide_source
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
            )

        if dataset == self.__source.dtype:
            new_source = self.__source.with_new_columns(
                **new_columns, descriptions=descriptions
            )
            return StructureCollection(
                new_source,
                self.__header,
                self.__datasets,
            )
        elif dataset not in self.__datasets.keys():
            raise ValueError(f"Dataset {dataset} not found in this collection!")

        ds = self.__datasets[dataset]

        if not isinstance(ds, oc.Dataset):
            raise ValueError(f"{dataset} is not a dataset!")

        new_ds = ds.with_new_columns(**new_columns, descriptions=descriptions)
        return StructureCollection(
            self.__source,
            self.__header,
            {**self.__datasets, dataset: new_ds},
            self.__hide_source,
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

        link_handler = None
        data_columns = set(self.__source.columns)
        metadata_columns: list[str] = reduce(
            lambda acc, key: acc + self.__handler.columns[key], data_types, []
        )
        rename_galaxies = "galaxies" in self.keys()
        for row in self.__source.rows(metadata_columns=metadata_columns):
            row = dict(row)
            links = self.__handler.parse(row)
            output = {
                key: self.__datasets[key].take_rows(index)
                for key, index in links.items()
                if key in data_types
            }
            if not output:
                continue
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
            self.__source, self.__header, new_datasets, hide_source
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

        return schema
