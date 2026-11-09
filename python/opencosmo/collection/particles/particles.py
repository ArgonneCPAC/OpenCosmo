from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from opencosmo.collection.particles import operations as pops
from opencosmo.collection.particles.state import ParticleState
from opencosmo.dataset.dataset import Dataset
from opencosmo.spatial.builders import make_box

if TYPE_CHECKING:
    from collections.abc import Callable, ItemsView, Iterable, Iterator, KeysView

    import astropy.units as u

    from opencosmo.dataset.dataset import OpenCosmoData
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region
    from opencosmo.spatial.region import Point3d


class ParticleCollection:
    """
    The particle types of a snapshot particle file (data type
    ``snapshot_particles``), such as dark matter, gas, star and AGN particles.

    Each particle type is a :py:class:`opencosmo.Dataset` with its own spatial
    index. The types are not linked to each other. The collection behaves like
    a read-only dictionary keyed by the particle type, for example
    ``collection["gas_particles"]``. :py:meth:`bound` and :py:meth:`with_units`
    apply to every particle type. :py:meth:`select` and :py:meth:`drop` match
    columns to the particle types that have them. For other operations,
    transform the datasets one at a time.
    """

    def __init__(self, state: ParticleState) -> None:
        self.__state = state

    def __with_state(self, state: ParticleState) -> ParticleCollection:
        if state is self.__state:
            return self
        return ParticleCollection(state)

    def keys(self) -> KeysView[str]:
        return self.__state.members.keys()

    def values(self) -> list[Dataset]:
        return [Dataset(member) for member in self.__state.members.values()]

    def items(self) -> ItemsView[str, Dataset]:
        return {
            name: Dataset(member) for name, member in self.__state.members.items()
        }.items()

    def __len__(self) -> int:
        return len(self.__state)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __getitem__(self, key: str) -> Dataset:
        if key not in self.__state.members:
            raise KeyError(
                f"Particle type {key} not found in collection. "
                f"Available types: {list(self.keys())}"
            )
        return Dataset(self.__state.members[key])

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_details: object) -> None:
        for child in self.values():
            child.__exit__(*exc_details)

    def __repr__(self) -> str:
        lengths = ", ".join(
            f"{name}: {len(member)}" for name, member in self.__state.members.items()
        )
        return f"ParticleCollection({lengths})"

    @property
    def header(self) -> OpenCosmoHeader:
        """
        The header of the file, which all particle types share.

        Returns
        -------
        header : opencosmo.header.OpenCosmoHeader
        """
        return self.__state.header

    @property
    def dtype(self) -> str:
        """
        The data type of the file, ``"snapshot_particles"``.

        Returns
        -------
        dtype : str
        """
        return self.__state.dtype

    def bound(self, region: Region, select_by: str | None = None) -> ParticleCollection:
        """
        Restrict every particle type to the particles inside a region. Each type
        is queried through its own spatial index, as in
        :py:meth:`opencosmo.Dataset.bound`.

        See :doc:`spatial_ref` for details of how to construct regions.

        Parameters
        ----------
        region : opencosmo.spatial.Region
            The region to query.
        select_by : str, optional
            The coordinates to query with. Particles have one set of
            coordinates, so this can be left out.

        Returns
        -------
        collection : opencosmo.ParticleCollection
            A new collection with the particles inside the region.
        """
        return self.__with_state(pops.bound(self.__state, region, select_by))

    def with_units(
        self,
        convention: str | None = None,
        conversions: dict[u.UnitBase, u.UnitBase] | None = None,
        **columns: u.UnitBase,
    ) -> ParticleCollection:
        """
        Change the unit convention of every particle type, convert all columns
        with a given unit into a different unit, and/or convert specific columns
        to a compatible unit. This behaves like
        :py:meth:`opencosmo.Dataset.with_units`, applied to each type. A column
        conversion applies to every type that has the column.

        Parameters
        ----------
        convention : str, optional
            The unit convention to use. One of "unitless", "scalefree",
            "comoving", or "physical".
        conversions : dict[astropy.units.Unit, astropy.units.Unit], optional
            Conversions that apply to all columns with the unit given by the key.
        **columns : astropy.units.Unit
            Conversions for columns with a specific name.

        Returns
        -------
        collection : opencosmo.ParticleCollection
            A new collection with the requested units.

        Raises
        ------
        ValueError
            If no particle type has one of the named columns.
        """
        if conversions is None:
            conversions = {}
        return self.__with_state(
            pops.with_units(self.__state, convention, conversions, **columns)
        )

    def box_search(self, p1: Point3d, p2: Point3d) -> ParticleCollection:
        """
        Restrict every particle type to the particles inside a 3D box.
        Equivalent to ``collection.bound(oc.make_box(p1, p2))``.

        Parameters
        ----------
        p1 : tuple[float, float, float]
            One corner of the box.
        p2 : tuple[float, float, float]
            The opposite corner of the box.

        Returns
        -------
        collection : opencosmo.ParticleCollection
            A new collection with the particles inside the box.
        """
        return self.bound(make_box(p1, p2))

    def select(
        self,
        *select_args: str | Iterable[str],
        mode: Literal["local", "global"] = "global",
        **select_kwargs: Any,
    ) -> ParticleCollection:
        """
        Select columns from the particle types in this collection.

        By default, column names and derived columns are matched to the
        particle types that have the columns they need. A name that several
        types have is selected from each of them:

        .. code-block:: python

            collection = collection.select("x", "y", "z", "sfr")

        This keeps the positions of every type, and ``sfr`` for the gas
        particles. Exact names must exist in at least one type. Wildcards apply
        to every type and leave types with no matches unchanged.

        To target particular types explicitly, use the type names as keyword
        arguments. A derived column is given with ``columns`` and
        ``derived_columns`` entries:

        .. code-block:: python

            collection = collection.select(
                dm_particles=["x", "y", "z"],
                gas_particles={
                    "columns": ["rho"],
                    "derived_columns": {"px": oc.col("mass") * oc.col("vx")},
                },
            )

        Parameters
        ----------
        *select_args : str or Iterable[str]
            Column names or wildcard patterns to match automatically across all
            particle types.
        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions nested inside derived column
            expressions are computed when running under MPI. Forwarded to each
            underlying :py:meth:`opencosmo.Dataset.select` call.
        **select_kwargs : Column or particle type selection
            Derived columns to route automatically, or explicit selections keyed
            by particle type.

        Returns
        -------
        collection : opencosmo.ParticleCollection
            A new collection with the selected columns. Particle types without
            an applicable automatic selection are unchanged.

        Raises
        ------
        ValueError
            If an exact column or derived-column dependency cannot be found in
            any particle type, an explicitly named type does not exist, or a
            scalar reduction is selected directly from the collection.
        """
        return self.__with_state(
            pops.select(self.__state, select_args, select_kwargs, mode)
        )

    def drop(
        self,
        *drop_args: str | Iterable[str],
        **columns_to_drop: str | Iterable[str],
    ) -> ParticleCollection:
        """
        Drop columns from the particle types in this collection. Like
        :py:meth:`select`, column names are matched to the types that have
        them, and wildcards apply to every type. To target types explicitly,
        pass type names as keyword arguments. To drop whole particle types, use
        :py:meth:`with_datasets`.

        Parameters
        ----------
        *drop_args : str or Iterable[str]
            Column names or wildcard patterns to match automatically across all
            particle types.
        **columns_to_drop : str or Iterable[str]
            Columns to drop from explicitly named particle types.

        Returns
        -------
        collection : opencosmo.ParticleCollection
            A new collection with the columns dropped.

        Raises
        ------
        ValueError
            If a column cannot be found in any particle type, or an explicitly
            named type does not exist.
        """
        return self.__with_state(pops.drop(self.__state, drop_args, columns_to_drop))

    def with_datasets(self, datasets: str | Iterable[str]) -> ParticleCollection:
        """
        Create a new collection with only some of the particle types.

        Parameters
        ----------
        datasets : str or Iterable[str]
            The particle types to keep.

        Returns
        -------
        collection : opencosmo.ParticleCollection

        Raises
        ------
        ValueError
            If a particle type is not in this collection, or no type is given.
        """
        return self.__with_state(pops.with_datasets(self.__state, datasets))

    def evaluate_on_dataset(
        self,
        func: Callable,
        dataset: str,
        vectorize: bool = True,
        format: str = "astropy",
        insert: bool = True,
        batch_size: int = -1,
        allow_overwrite: bool = False,
        **evaluate_kwargs: Any,
    ) -> ParticleCollection | dict[str, Any] | None:
        """
        Evaluate a function on one particle type in this collection. This is
        :py:meth:`opencosmo.Dataset.evaluate` applied to that type, with the
        result kept in the collection when ``insert`` is true.

        Parameters
        ----------
        func : Callable
            The function to evaluate on the rows of the particle type.
        dataset : str
            The particle type to evaluate on, for example ``"gas_particles"``.
        vectorize : bool, default = True
            Whether to provide the values as full columns (True) or one row at
            a time (False). Ignored if ``batch_size`` is set.
        format : str, default = "astropy"
            The format in which to provide column data to the function. Supports
            the same formats as :py:meth:`opencosmo.Dataset.get_data`.
        insert : bool, default = True
            If true, the output is inserted as new columns of the particle type
            and a new collection is returned. Otherwise the output is returned
            directly.
        batch_size : int, default = -1
            If set, feed data to the function in batches of this size. The
            default of -1 disables batching.
        allow_overwrite : bool, default = False
            Whether the output may replace existing columns.
        **evaluate_kwargs : Any
            Additional arguments for the function, passed as keyword arguments.
            An array with the same length as the particle type is treated as an
            additional column.

        Returns
        -------
        result : opencosmo.ParticleCollection or dict or None
            The new collection if ``insert`` is true, otherwise the output of
            the function.

        Raises
        ------
        ValueError
            If the particle type is not in this collection.
        """
        result = pops.evaluate_on_dataset(
            self.__state,
            func,
            dataset,
            vectorize,
            format,
            insert,
            batch_size,
            allow_overwrite,
            **evaluate_kwargs,
        )
        if isinstance(result, ParticleState):
            return self.__with_state(result)
        return result

    def get_data(
        self,
        format: str = "astropy",
        unpack: bool = True,
        wrap_single: bool = False,
        **kwargs: Any,
    ) -> dict[str, OpenCosmoData]:
        """
        Get the data of every particle type, keyed by particle type. The
        arguments are passed to :py:meth:`opencosmo.Dataset.get_data`.

        Parameters
        ----------
        format : str, default = "astropy"
            The output format, as in :py:meth:`opencosmo.Dataset.get_data`.
        unpack : bool, default = True
            As in :py:meth:`opencosmo.Dataset.get_data`.
        wrap_single : bool, default = False
            As in :py:meth:`opencosmo.Dataset.get_data`.

        Returns
        -------
        data : dict[str, Any]
            The data of each particle type, keyed by particle type.
        """
        return {
            name: dataset.get_data(format, unpack, wrap_single, **kwargs)
            for name, dataset in self.items()
        }

    def make_schema(self, path: str) -> Schema:
        return pops.make_schema(self.__state, path)
