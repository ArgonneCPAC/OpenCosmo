from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from opencosmo.collection.lightcone.scope import LightconeScope

if TYPE_CHECKING:
    from collections.abc import Iterable
    from uuid import UUID

    import astropy.units as u

    from opencosmo.collection.lightcone.healpix_map import HealpixMap
    from opencosmo.dataset.state import DatasetState
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.spatial.protocols import Region


LeafKey = tuple[int | str, ...]


@dataclass(frozen=True)
class LightconeState:
    """
    Backend state for a Lightcone. Leaves are stored flat and in stacking order,
    keyed by path: ``(step,)`` for ordinary lightcones, ``(step, subtype)`` for
    lightcones with several datasets per step. Construct with ``from_leaves``.
    """

    leaves: tuple[tuple[LeafKey, DatasetState], ...]
    header: OpenCosmoHeader
    maps: HealpixMap | None = None
    hidden: frozenset[str] = frozenset()
    sort_key: tuple[str, bool] | None = None
    scope: LightconeScope = field(default_factory=LightconeScope)

    def __len__(self) -> int:
        return sum(len(leaf) for _, leaf in self.leaves)

    @property
    def z_range(self) -> tuple[float, float]:
        return self.header.lightcone["z_range"]

    @property
    def uuid(self) -> UUID:
        return self.leaves[0][1].uuid

    @property
    def columns(self) -> list[str]:
        columns = [c for c in self.leaves[0][1].columns if c not in self.hidden]
        columns.extend(name for name in self.scope.names() if name not in columns)
        return columns

    @property
    def descriptions(self) -> dict[str, str | None]:
        descriptions = self.leaves[0][1].descriptions
        return {k: v for k, v in descriptions.items() if k not in self.hidden}

    @property
    def units(self) -> dict[str, u.Unit | None]:
        units = self.leaves[0][1].units
        return {k: v for k, v in units.items() if k not in self.hidden}

    @property
    def region(self) -> Region | None:
        regions = [leaf.region for _, leaf in self.leaves]
        if len(regions) == 1:
            return regions[0]
        return regions[0].combine(*regions[1:])  # type: ignore[union-attr]


def from_leaves(
    leaves: Iterable[tuple[LeafKey, DatasetState]],
    maps: HealpixMap | None = None,
    z_range: tuple[float, float] | None = None,
    hidden: frozenset[str] = frozenset(),
    sort_key: tuple[str, bool] | None = None,
    scope: LightconeScope | None = None,
) -> LightconeState:
    leaves = tuple(leaves)
    if not leaves:
        raise ValueError("A lightcone must contain at least one dataset!")
    if len({len(key) for key, _ in leaves}) != 1:
        raise ValueError("All lightcone leaf keys must have the same depth!")
    if len({frozenset(leaf.columns) for _, leaf in leaves}) != 1:
        raise ValueError("Not all lightcone datasets have the same columns!")

    if z_range is None:
        leaf_ranges = [header_z_range(leaf.header) for _, leaf in leaves]
        z_range = (
            min(zr[0] for zr in leaf_ranges),
            max(zr[1] for zr in leaf_ranges),
        )
    header = leaves[0][1].header.with_parameter("lightcone/z_range", z_range)

    return LightconeState(
        leaves,
        header,
        maps,
        hidden,
        sort_key,
        scope if scope is not None else LightconeScope(),
    )


def header_z_range(header: OpenCosmoHeader) -> tuple[float, float]:
    """The redshift range covered by a single lightcone dataset."""
    z_range = header.lightcone["z_range"]
    if z_range is not None:
        return z_range
    step = header.file.step
    assert step is not None
    step_zs = header.simulation["step_zs"]
    return (step_zs[step], step_zs[step - 1])


def public_keys(state: LightconeState) -> list[int | str]:
    """The keys a user sees: the first key component, in stacking order."""
    return list(dict.fromkeys(key[0] for key, _ in state.leaves))


def view(state: LightconeState, key: int | str) -> DatasetState | LightconeState:
    """
    The value a user sees for ``key``: the leaf itself if the key names exactly
    one leaf, otherwise a lightcone over the matching leaves with ``key``
    stripped. Views carry no maps, hidden columns, sort, or scope.
    """
    group = tuple((k[1:], leaf) for k, leaf in state.leaves if k[0] == key)
    if not group:
        raise KeyError(key)
    if len(group) == 1 and group[0][0] == ():
        return group[0][1]
    return from_leaves(group)
