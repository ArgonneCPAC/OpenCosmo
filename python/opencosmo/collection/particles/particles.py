from __future__ import annotations

from typing import TYPE_CHECKING, Self

from opencosmo.collection.particles import operations as pops
from opencosmo.dataset.dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView

    import astropy.units as u

    from opencosmo.collection.particles.state import ParticleState
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.spatial.protocols import Region


class ParticleCollection:
    """
    The particle types of a snapshot particle file (data type
    ``snapshot_particles``), such as dark matter, gas, star and AGN particles.

    Each particle type is a :py:class:`opencosmo.Dataset` with its own spatial
    index. The types are not linked to each other. The collection behaves like
    a read-only dictionary keyed by the particle type, for example
    ``collection["gas_particles"]``. :py:meth:`bound` and :py:meth:`with_units`
    apply to every particle type. For other operations, transform the datasets
    one at a time.
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
