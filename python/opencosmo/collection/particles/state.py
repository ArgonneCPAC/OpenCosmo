from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opencosmo.dataset.state import DatasetState
    from opencosmo.header import OpenCosmoHeader


@dataclass(frozen=True)
class ParticleState:
    """
    Backend state for a ParticleCollection: the particle types of one snapshot
    particle file, keyed by group name. The types share the file's header and
    are not linked to each other.
    """

    members: Mapping[str, DatasetState]

    def __post_init__(self) -> None:
        if not self.members:
            raise RuntimeError("A particle collection needs at least one member")

    def __len__(self) -> int:
        return len(self.members)

    @property
    def header(self) -> OpenCosmoHeader:
        return next(iter(self.members.values())).header

    @property
    def dtype(self) -> str:
        return str(self.header.file.data_type)
