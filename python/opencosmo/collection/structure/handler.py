from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from uuid import UUID

    import opencosmo as oc
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
