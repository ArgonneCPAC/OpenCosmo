from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.structure import links
from opencosmo.collection.structure import structure as sc
from opencosmo.dataset import dataset as ocds
from opencosmo.dataset.state import DatasetState

if TYPE_CHECKING:
    from uuid import UUID

    import opencosmo as oc
    from opencosmo.io.schema import Schema
    from opencosmo.mapping.mapping import DatasetMatchSet

    Child = oc.Dataset | oc.Lightcone | oc.StructureCollection


"""
A tale in 3 acts:

Act 1: Patrick creates a LinkHandler which holds a pointer to the index that determines
which rows in halo/galaxy properties corresponds to rows in particle datasets.

Act 2: Patrick unifies metadata handling in Datasets, making the link handler
unncessary. Instead, ephemeral particle datasets are created when requested.

Act 3: Patrick realizes this solution makes it impossible to cache things, particularly
very expensive computations that the user has created with evaluate. Patrick re-introduces
the LinkHandler, but it's better this time or something.

"""

LINK_ALIASES = {  # Left: Name in file, right: Name in collection
    "sodbighaloparticles_star_particles": "star_particles",
    "sodbighaloparticles_dm_particles": "dm_particles",
    "sodbighaloparticles_gravity_particles": "gravity_particles",
    "sodbighaloparticles_agn_particles": "agn_particles",
    "sodbighaloparticles_gas_particles": "gas_particles",
    "fofbighaloparticles_star_particles": "star_particles",
    "fofbighaloparticles_dm_particles": "dm_particles",
    "fofbighaloparticles_gravity_particles": "gravity_particles",
    "fofbighaloparticles_agn_particles": "agn_particles",
    "fofbighaloparticles_gas_particles": "gas_particles",
    "sod_profile": "halo_profiles",
    "haloprofiles": "halo_profiles",
    "galaxyproperties": "galaxy_properties",
    "galaxyparticles_star_particles": "star_particles",
}


class LinkHandler:
    """Manage linked structure datasets and their deferred rebuilding."""

    def __init__(
        self,
        match_sets: dict[UUID, DatasetMatchSet],
        derived_from: Optional[oc.Dataset | oc.Lightcone],
    ) -> None:
        self.__derived_from = derived_from
        self.match_sets = match_sets

    @property
    def derived_from(self) -> Optional[oc.Dataset | oc.Lightcone]:
        """The source the linked datasets are currently aligned with, if not the source."""
        return self.__derived_from

    def make_schema(
        self, source: oc.Dataset | oc.Lightcone, source_schema: Schema
    ) -> Schema:
        """Add source-row-aligned link datasets to a structure source schema."""
        return links.link_schema(self.match_sets, source._state, source_schema)

    @property
    def names(self) -> tuple[str, ...]:
        """Collection-facing link names, sorted for determinism."""
        return links.link_names(self.match_sets)

    def make_derived(self, source: oc.Dataset | oc.Lightcone) -> LinkHandler:
        """Record the source from which deferred linked rebuilding begins."""
        derived_from = self.__derived_from
        if self.__derived_from is None:
            derived_from = source
        return LinkHandler(self.match_sets, derived_from)

    def resort(
        self,
        source: oc.Dataset | oc.Lightcone,
        datasets: Mapping[str, Child],
    ) -> dict[str, Child]:
        """Restore linked datasets to their source's original file order."""
        targets = LinkHandler.__unwrap(datasets)
        return LinkHandler.__wrap(links.resort(self.match_sets, source._state, targets))

    @staticmethod
    def __unwrap(datasets: Mapping[str, Child]) -> dict[str, links.LinkTarget]:
        return {name: dataset._state for name, dataset in datasets.items()}

    @staticmethod
    def __wrap(targets: Mapping[str, links.LinkTarget]) -> dict[str, Child]:
        wrapped: dict[str, Child] = {}
        for name, target in targets.items():
            if isinstance(target, DatasetState):
                wrapped[name] = ocds.Dataset(target)
            elif isinstance(target, LightconeState):
                wrapped[name] = lc.Lightcone(target)
            else:
                wrapped[name] = sc.StructureCollection._from_state(target)
        return wrapped
