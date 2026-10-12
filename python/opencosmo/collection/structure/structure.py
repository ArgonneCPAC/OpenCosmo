from __future__ import annotations

from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    Optional,
    cast,
)

import numpy as np

import opencosmo as oc
from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.structure import io as sio
from opencosmo.collection.structure import operations as scops
from opencosmo.collection.structure import state as scst
from opencosmo.collection.structure.io import rebuild_data_linked
from opencosmo.dataset.state import DatasetState
from opencosmo.io.schema import FileEntry, make_schema

from .handler import LinkHandler

if TYPE_CHECKING:
    import astropy.units as u

    from opencosmo.column.column import ConstructedColumn
    from opencosmo.dtypes import HaccSimulationParameters
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.mpi import MPI
    from opencosmo.spatial.protocols import Region


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
        source: oc.Dataset | oc.Lightcone,
        datasets: Mapping[str, oc.Dataset | oc.Lightcone | StructureCollection],
        hide_source: bool = False,
        link_handler: Optional[LinkHandler] = None,
        derived_columns: Optional[set[str]] = None,
        *,
        resolve_links: bool = False,
        **kwargs,
    ):
        """
        Initialize a linked collection with the provided datasets and links.
        """
        if link_handler is None:
            raise ValueError("Structure collections require a link handler")

        state = scst.from_parts(
            source._state,
            {name: dataset._state for name, dataset in datasets.items()},
            link_handler.match_sets,
            resolve=resolve_links,
            hide_source=hide_source,
            derived_columns=frozenset(derived_columns or ()),
        )
        if link_handler.derived_from is not None:
            state = replace(state, linked_source=link_handler.derived_from._state)
        self.__state = state

    @classmethod
    def _from_state(cls, state: scst.StructureState) -> StructureCollection:
        collection = cls.__new__(cls)
        collection.__state = state
        return collection

    @property
    def _state(self) -> scst.StructureState:
        """Return the internal state for collection and I/O implementations."""
        return self.__state

    def __with_state(self, state: scst.StructureState) -> StructureCollection:
        if state is self.__state:
            return self
        return StructureCollection._from_state(state)

    @staticmethod
    def __wrap(
        child: scst.StructureChild,
    ) -> oc.Dataset | oc.Lightcone | StructureCollection:
        if isinstance(child, DatasetState):
            return oc.Dataset(child)
        if isinstance(child, scst.StructureState):
            return StructureCollection._from_state(child)
        return lc.Lightcone(child)

    # Transitional views of the state in the shape of the old attributes. They
    # go away as the operations move onto StructureState.

    @property
    def __source(self) -> oc.Dataset:
        # Typed as the old attribute was; lightcone sources are Lightcones.
        return cast("oc.Dataset", StructureCollection.__wrap(self.__state.source))

    @property
    def __handler(self) -> LinkHandler:
        linked_source = self.__state.linked_source
        derived_from: oc.Dataset | oc.Lightcone | None = None
        if linked_source is not None:
            wrapped = StructureCollection.__wrap(linked_source)
            assert not isinstance(wrapped, StructureCollection)
            derived_from = wrapped
        return LinkHandler(dict(self.__state.match_sets), derived_from)

    def __get_datasets(self) -> dict[str, oc.Dataset | StructureCollection]:
        """
        The linked datasets, rebuilt to match the current source.
        """
        return StructureCollection.__views(self.__state.children)

    @staticmethod
    def __views(
        children: Mapping[str, scst.StructureChild],
    ) -> dict[str, oc.Dataset | StructureCollection]:
        # Typed as the old attributes were; lightcone children are Lightcones.
        return cast(
            "dict[str, oc.Dataset | StructureCollection]",
            {
                name: StructureCollection.__wrap(child)
                for name, child in children.items()
            },
        )

    def __repr__(self):
        structure_type = self.__source.header.file.data_type.split("_")[0] + "s"
        is_lightcone = isinstance(self.__source, lc.Lightcone)
        keys = list(self.keys())
        if len(keys) == 2:
            dtype_str = " and ".join(keys)
        else:
            dtype_str = ", ".join(keys[:-1]) + ", and " + keys[-1]
        header = f"Collection of {structure_type} {'on a lightcone ' if is_lightcone else ' '}with {dtype_str}\n"
        source_repr = self.__source.__repr__().split("\n", maxsplit=1)[1]

        return header + source_repr

    def __len__(self):
        return len(self.__state)

    def __dir__(self):
        return list(self.__state.header.parameters.keys()) + super().__dir__()

    def __getattr__(self, key: str):
        if key == "_StructureCollection__state":
            raise AttributeError(key)
        try:
            return self.__state.header.parameters[key]
        except KeyError:
            return object.__getattribute__(self, key)

    @classmethod
    def open(
        cls,
        targets: list[DatasetTarget],
        ignore_empty=True,
        index_kind: str = "none",
        is_empty_ref: bool = False,
        **kwargs,
    ) -> StructureCollection:
        return sio.build_structure_collection(
            targets,
            ignore_empty,
            index_kind=index_kind,
            is_empty_ref=is_empty_ref,
        )

    @property
    def dtype(self):
        structure_type = self.__state.header.file.dt
        return structure_type

    @property
    def header(self) -> OpenCosmoHeader:
        return self.__state.header

    @property
    def properties(self) -> list[str]:
        """
        The high-level properties that are available as part of the
        halo_properties or galaxy_properties dataset.
        """
        return self.__state.properties

    @property
    def sorted_by(self) -> Optional[str]:
        """
        The column this collection is currently sorted by, or ``None`` if unsorted.

        Returns
        -------
        column : Optional[str]
        """
        return self.__state.sorted_by

    @property
    def simulation(self) -> HaccSimulationParameters | None:
        """
        Get the parameters of the simulation this dataset is drawn
        from.

        Returns
        -------
        parameters: opencosmo.dtypes.HaccSimulationParameters
        """
        return self.__source.simulation

    def keys(self) -> list[str]:
        """
        Return the names of the datasets in this collection.
        """
        return scst.public_keys(self.__state)

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
        return cast(
            "oc.Dataset | StructureCollection",
            StructureCollection.__wrap(scst.view(self.__state, key)),
        )

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
        return self.__state.region

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
        return self.__with_state(scops.bound(self.__state, region, select_by))

    def with_redshift_range(self, z_low: float, z_high: float) -> StructureCollection:
        """
        Restrict this lightcone structure collection to a specific redshift range.
        This is more efficient than filtering on the redshift column directly because
        it prunes entire redshift steps before row-level filtering, and it updates
        the z_range metadata on the returned collection.

        Parameters
        ----------
        z_low : float
            The lower bound of the redshift range (inclusive).
        z_high : float
            The upper bound of the redshift range (inclusive).

        Returns
        -------
        result : StructureCollection
            A new StructureCollection restricted to the given redshift range.

        Raises
        ------
        AttributeError
            If this is not a lightcone structure collection.
        ValueError
            If the requested range does not overlap the available redshift range.
        """
        return self.__with_state(scops.with_redshift_range(self.__state, z_low, z_high))

    def cone_search(self, center, radius) -> StructureCollection:
        """
        Search for structures within an angular distance of a point on the sky.
        Equivalent to ``collection.bound(oc.make_cone(center, radius))``.

        Parameters
        ----------
        center : tuple | astropy.coordinates.SkyCoord
            Center of the search cone. If a tuple with no units, assumed to be
            (RA, Dec) in degrees.
        radius : float | astropy.units.Quantity
            Angular radius of the search cone. If no units, assumed to be degrees.

        Returns
        -------
        result : StructureCollection

        Raises
        ------
        AttributeError
            If this is not a lightcone structure collection.
        """
        scops.require_lightcone(self.__state, "cone_search")
        return self.bound(oc.make_cone(center, radius))

    def box_search(self, p1, p2) -> StructureCollection:
        """
        Search for structures within a rectangular region of the sky (defined by
        RA/Dec corners). Equivalent to ``collection.bound(oc.make_skybox(p1, p2))``.

        Parameters
        ----------
        p1 : tuple | astropy.coordinates.SkyCoord
            One corner of the box. If a tuple with no units, assumed to be
            (RA, Dec) in degrees.
        p2 : tuple | astropy.coordinates.SkyCoord
            The opposite corner of the box.

        Returns
        -------
        result : StructureCollection

        Raises
        ------
        AttributeError
            If this is not a lightcone structure collection.
        """
        scops.require_lightcone(self.__state, "box_search")
        return self.bound(oc.make_skybox(p1, p2))

    def get_pixels(self, nside: int = 64) -> np.ndarray:
        """
        Return the HEALPix pixels occupied by this lightcone structure collection
        at a given resolution.

        Pixel indices are returned in nested ordering. The ``nside`` parameter
        controls angular resolution: larger values produce finer pixels. The
        requested resolution may not exceed the resolution of the spatial index
        stored in the file.

        Parameters
        ----------
        nside : int, default = 64
            The HEALPix resolution parameter. Must be a positive power of two.

        Returns
        -------
        pixels : numpy.ndarray[int]
            HEALPix pixel indices (nested ordering) occupied by this collection
            at the given resolution.

        Raises
        ------
        AttributeError
            If this is not a lightcone structure collection.
        ValueError
            If ``nside`` is not a positive power of two, if ``nside`` exceeds
            the maximum resolution of the spatial index, or if the lightcone
            does not have a spatial index.
        """
        return scops.get_pixels(self.__state, nside)

    def pixel_search(self, pixels: np.ndarray, nside: int = 64) -> StructureCollection:
        """
        Return the subset of this lightcone structure collection that falls within
        a set of HEALPix pixels.

        Pixels must be specified in nested ordering and must be valid indices at
        the given ``nside``. Duplicate pixel indices are ignored. Use
        :py:meth:`get_pixels <opencosmo.StructureCollection.get_pixels>` to
        discover which pixels this collection covers.

        Parameters
        ----------
        pixels : array_like[int]
            HEALPix pixel indices to query, in nested ordering. Must be a 1-D
            array of non-negative integers. Values must be less than
            ``healpy.nside2npix(nside)``.
        nside : int, default = 64
            The HEALPix resolution parameter. Must be a positive power of two
            and must not exceed the resolution of the spatial index stored in
            the file.

        Returns
        -------
        result : StructureCollection
            A new collection containing only the structures that fall within
            the specified pixels.

        Raises
        ------
        AttributeError
            If this is not a lightcone structure collection.
        ValueError
            If ``nside`` is not a positive power of two, or if ``pixels``
            contains values that are out of range for the given ``nside``.
        """
        return self.__with_state(scops.pixel_search(self.__state, pixels, nside))

    def evaluate(
        self,
        func: Callable,
        dataset: Optional[str] = None,
        format: str = "astropy",
        insert: bool = True,
        allow_overwrite: bool = False,
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
            The format in which to provide column data to your function. Supports the same formats
            as :py:meth:`get_data <opencosmo.Dataset.get_data>` ("astropy", "numpy", "pandas",
            "polars", "arrow", "jax"). When :code:`insert=True`, the function's output is converted
            back to numpy before being stored.

        **evaluate_kwargs: any,
            Any additional arguments that are required for your function to run. These will be passed directly
            to the function as keyword arguments. If a kwarg is an array of values with the same length as the dataset,
            it will be treated as an additional column.

        """
        result = scops.evaluate(
            self.__state,
            func,
            dataset,
            format,
            insert,
            allow_overwrite,
            **evaluate_kwargs,
        )
        if isinstance(result, scst.StructureState):
            return self.__with_state(result)
        return result

    def evaluate_on_dataset(
        self,
        func: Callable,
        dataset: Optional[str] = None,
        vectorize: bool = False,
        format: str = "astropy",
        insert: bool = True,
        batch_size: int = -1,
        allow_overwrite: bool = False,
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

        Parameters
        ----------

        func: Callable
            The function to evaluate on the rows in the dataset.

        dataset: Optional[str] = None,
            The dataset to perform the evaluation on. If None, defaults to the halo_properties or galaxy_properties dataset.

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False). Ignored if :code:`batch_size` is set.

        format: str, default = astropy
            The format in which to provide column data to your function. Supports the same formats
            as :py:meth:`get_data <opencosmo.Dataset.get_data>` ("astropy", "numpy", "pandas",
            "polars", "arrow", "jax"). When :code:`insert=True`, the function's output is converted
            back to numpy before being stored.

        insert: bool, default = True
            If true, the data will be inserted as a column in this dataset. The new column will have the same name
            as the function. Otherwise the data will be returned directly.

        batch_size: int, default = -1
            If set, feed data to the function in batches of the specified size. Default is -1, which disables batching. If
            set to another value, the :code:`vectorize` flag is ignored.

        **evaluate_kwargs: any,
            Any additional arguments that are required for your function to run. These will be passed directly
            to the function as keyword arguments. If a kwarg is an array of values with the same length as the dataset,
            it will be treated as an additional column.


        """
        result = scops.evaluate_on_dataset(
            self.__state,
            func,
            dataset,
            vectorize,
            format,
            insert,
            batch_size=batch_size,
            allow_overwrite=allow_overwrite,
            **evaluate_kwargs,
        )
        if isinstance(result, scst.StructureState):
            return self.__with_state(result)
        return result

    def filter(
        self, *masks, on_galaxies: bool = False, mode: str = "global"
    ) -> StructureCollection:
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

        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions inside the masks are computed when
            running under MPI. Defaults to ``"global"`` (cross-rank); pass
            ``"local"`` for per-rank scalars. Forwarded to the underlying
            ``Dataset.filter`` call.

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
        return self.__with_state(
            scops.filter(self.__state, *masks, on_galaxies=on_galaxies, mode=mode)
        )

    def select(
        self,
        *select_args,
        mode: str = "global",
        **select_kwargs: str | Iterable[str] | dict,
    ) -> StructureCollection:
        """
        Select columns from the datasets in this collection.

        By default, column names and derived columns are matched to the datasets
        that contain their required columns. This works across nested structure
        collections, so dataset names do not usually need to be specified:

        .. code-block:: python

            momentum = oc.col("gal_mass_star") * oc.col("gal_com_vx")
            collection = collection.select(
                "fof_halo_mass",
                "gal_mass_star",
                gal_momentum=momentum,
            )

        Exact names must exist in at least one dataset. Wildcards are applied to
        every dataset and leave datasets with no matches unchanged. A column or
        derived expression that can be resolved by multiple datasets is selected
        from each of them.

        To target particular datasets explicitly, use dataset names as keyword
        arguments. This legacy form is useful when the same column name appears in
        several datasets. Nested collections and derived columns can also be targeted
        with dictionaries:

        .. code-block:: python

            collection = collection.select(
                halo_properties=["fof_halo_mass", "sod_halo_mass"],
                dm_particles=["x", "y", "z"],
                galaxies={"galaxy_properties": ["gal_mass_star"]},
            )

        In the explicit form, a derived selection is represented by ``columns`` and
        ``derived_columns`` entries:

        .. code-block:: python

            collection = collection.select(
                halo_properties={
                    "columns": ["fof_halo_mass"],
                    "derived_columns": {
                        "fof_halo_px": oc.col("fof_halo_mass")
                        * oc.col("fof_halo_com_vx")
                    },
                },
            )

        Parameters
        ----------
        *select_args : str or Iterable[str]
            Column names or wildcard patterns to match automatically across all
            datasets in the collection.
        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions nested inside derived column
            expressions are computed when running under MPI. Defaults to
            ``"global"`` (cross-rank); pass ``"local"`` for per-rank scalars.
            Forwarded to each underlying ``Dataset.select`` call.
        **select_kwargs : Column or dataset selection
            Derived columns to route automatically, or explicit selections keyed
            by dataset name. Explicit selections may be a column name, an iterable
            of names, or a nested selection dictionary.

        Returns
        -------
        StructureCollection
            A new collection containing the selected columns. Datasets without an
            applicable automatic selection are unchanged.

        Raises
        -------
        ValueError
            If an exact column or derived-column dependency cannot be found in any
            dataset, an explicitly named dataset does not exist, or a scalar
            reduction is selected directly from the collection.
        """
        return self.__with_state(
            scops.select(self.__state, *select_args, mode=mode, **select_kwargs)
        )

    def drop(
        self,
        *drop_args: str | Iterable[str],
        **columns_to_drop: str | Iterable[str] | dict,
    ):
        """
        Drop columns by automatically matching their names to datasets in this
        collection. Like :py:meth:`StructureCollection.select
        <opencosmo.StructureCollection.select>`, this works across nested
        structure collections; wildcards are applied to every dataset, while
        datasets without a match are unchanged.

        To target datasets explicitly, pass dataset names as keyword arguments.
        Values may be a string, iterable of strings, or nested dictionary.

        Datasets that are not included will not be modified. You can drop
        entire datasets with :py:meth:`with_datasets <opencosmo.StructureCollection.with_datasets>`

        Parameters
        ----------
        *drop_args : str or Iterable[str]
            Column names or wildcard patterns to match automatically across all
            datasets in the collection.
        **columns_to_drop : str | Iterable[str]
            Columns to drop from explicitly named datasets.

        Returns
        -------
        StructureCollection
            A new collection with the specified columns dropped.

        Raises
        -------
        ValueError
            If the specified dataset is not found in the collection.
        """
        return self.__with_state(
            scops.drop(self.__state, *drop_args, **columns_to_drop)
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
        return self.__with_state(scops.sort_by(self.__state, column, invert))

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
        return self.__with_state(
            scops.with_units(
                self.__state, convention, conversions, **dataset_conversions
            )
        )

    def take(
        self, n: int, at: str = "random", mode: Literal["local", "global"] = "local"
    ):
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
        StructureCollection
            A new collection with the structures taken from the original.
        """
        return self.__with_state(scops.take(self.__state, n, at, mode))

    def take_range(
        self, start: int, end: int, mode: Literal["local", "global"] = "local"
    ):
        """
        Create a new collection from a row range in this collection. We use standard
        indexing conventions, so the rows included will be start -> end - 1.

        Parameters
        ----------
        start : int
            The first row to get.
        end : int
            The last row to get.
        mode : str, "local" or "global", default = "local"
            Controls how ``start`` and ``end`` are interpreted when running
            under MPI. Has no effect if you are not using MPI.

            * ``"local"`` (default): the range is applied independently on
              each rank.
            * ``"global"``: ``start`` and ``end`` index into the global row
              space across all ranks combined. Each rank receives the portion
              of that range it owns. If the collection is sorted, ranks will
              coordinate to take from the globally-sorted collection.

        Returns
        -------
        collection : StructureCollection
            The collection with only the rows from start to end.

        Raises
        ------
        ValueError
            If start or end are negative or greater than the length of the dataset
            or if end is greater than start.

        """
        return self.__with_state(scops.take_range(self.__state, start, end, mode))

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
        return self.__with_state(scops.take_rows(self.__state, rows))

    def with_new_columns(
        self,
        dataset: str,
        descriptions: str | dict[str, str] = {},
        allow_overwrite: bool = False,
        mode: str = "global",
        **new_columns: ConstructedColumn | np.ndarray,
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

        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions nested inside derived column
            expressions are computed when running under MPI. Defaults to
            ``"global"`` (cross-rank); pass ``"local"`` for per-rank scalars.
            Forwarded to the underlying ``Dataset.with_new_columns`` call.

        ** columns: opencosmo.Column
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
        return self.__with_state(
            scops.with_new_columns(
                self.__state,
                dataset,
                descriptions,
                allow_overwrite,
                mode,
                **new_columns,
            )
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
        for structure in scops.iter_objects(self.__state, data_types):
            yield {
                name: value
                if isinstance(value, dict)
                else StructureCollection.__wrap(value)
                for name, value in structure.items()
            }

    def with_datasets(self, datasets: str | Iterable[str]):
        """
        Create a new collection out of a subset of the datasets in this collection.
        It is also possible to do this when you iterate over the collection with
        :py:meth:`StructureCollection.objects <opencosmo.StructureCollection.objects>`,
        however doing it up front may be more desirable if you don't plan to use
        the dropped datasets at any point.
        """
        return self.__with_state(scops.with_datasets(self.__state, datasets))

    def halos(self, *args, **kwargs):
        """
        Alias for "objects" in the case that this StructureCollection contains halos.
        """
        if self.__state.dtype == "halo_properties":
            yield from self.objects(*args, **kwargs)
        else:
            raise AttributeError("This collection does not contain halos!")

    def galaxies(self, *args, **kwargs):
        """
        Alias for "objects" in the case that this StructureCollection contains galaxies
        """
        if self.__state.dtype == "galaxy_properties":
            yield from self.objects(*args, **kwargs)
        else:
            raise AttributeError("This collection does not contain galaxies!")

    def make_schema(self, path: str, **kwargs) -> Schema:
        children = {}
        source_name = self.__source.dtype
        datasets = self.__handler.resort(self.__source, self.__get_datasets())
        schema_kwargs: dict[str, Any] = (
            {"no_stack": True} if isinstance(self.__source, lc.Lightcone) else {}
        )

        source_schema = self.__source.make_schema(
            **schema_kwargs, path="/".join([path, source_name])
        )
        if len(self.__source) > 0:
            children[source_name] = source_schema = self.__handler.make_schema(
                self.__source, source_schema
            )
            children[source_name] = rebuild_data_linked(source_schema)

        for name, dataset in datasets.items():
            if name == "galaxies":
                name = "galaxy_properties"
            ds_schema = dataset.make_schema(
                **schema_kwargs, path="/".join([path, name])
            )
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
