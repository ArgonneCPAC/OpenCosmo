from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Union

if TYPE_CHECKING:
    import h5py

    from opencosmo.collection.protocols import Collection
    from opencosmo.dataset import Dataset
    from opencosmo.io.iopen import DatasetTarget, FileTarget


@dataclass
class TargetSummary:
    """Summary of file targets for dispatch to appropriate collection opener.

    Built once during discovery (no file I/O), used to determine which
    collection class should open the targets.
    """

    dataset_targets: list[DatasetTarget] = field(default_factory=list)
    dataset_groups: dict[str, list[DatasetTarget]] = field(default_factory=dict)
    data_types: set[str] = field(default_factory=set)
    is_lightcone: set[bool] = field(default_factory=set)
    total_targets: int = 0
    has_link_targets: bool = False
    has_properties: bool = False
    parent_group_count: int = 0

    @classmethod
    def build(cls, targets: list[FileTarget]) -> TargetSummary:
        """Build a summary from discovery phase targets."""
        dataset_targets: list[DatasetTarget] = []
        dataset_groups: dict[str, list[DatasetTarget]] = {}
        data_types: set[str] = set()
        is_lightcone: set[bool] = set()
        has_link_targets = False
        has_properties = False
        parent_groups: set[h5py.Group] = set()

        for target in targets:
            dataset_targets.extend(target["dataset_targets"])
            # Merge dataset_groups
            for group_name, group_targets in target["dataset_groups"].items():
                if group_name not in dataset_groups:
                    dataset_groups[group_name] = []
                dataset_groups[group_name].extend(group_targets)

        # Collect all dataset targets (both from dataset_targets and dataset_groups)
        all_dataset_targets = list(dataset_targets)
        for group_targets in dataset_groups.values():
            all_dataset_targets.extend(group_targets)

        # Extract data types and lightcone status
        for dt in all_dataset_targets:
            data_types.add(str(dt["header"].file.data_type))
            is_lightcone.add(dt["header"].file.is_lightcone)
            parent_groups.add(dt["dataset_group"].parent)
            # Check if this is a properties dataset
            if dt["header"].file.data_type in [
                "halo_properties",
                "galaxy_properties",
            ]:
                has_properties = True

        has_particles = any("particle" in dt for dt in data_types)
        has_profiles = "halo_profiles" in data_types or "galaxy_profiles" in data_types

        if has_particles or has_profiles:
            has_link_targets = True

        return cls(
            dataset_targets=dataset_targets,
            dataset_groups=dataset_groups,
            data_types=data_types,
            is_lightcone=is_lightcone,
            total_targets=len(targets),
            has_link_targets=has_link_targets,
            has_properties=has_properties,
            parent_group_count=len(parent_groups),
        )


class Opener(Protocol):
    """Protocol for collection openers."""

    @classmethod
    def claim(cls, summary: TargetSummary) -> bool:
        """Return True if this opener can handle the targets."""
        ...

    @classmethod
    def open(
        cls, targets: list[FileTarget], **kwargs: bool
    ) -> Union[Dataset, Collection]:
        """Open the targets and return a Dataset or Collection."""
        ...
