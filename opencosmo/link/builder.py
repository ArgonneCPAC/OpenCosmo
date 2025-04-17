from __future__ import annotations

from typing import Iterable, Optional, Protocol, Self

from h5py import File, Group

import opencosmo as oc
from opencosmo.dataset.column import get_column_builders
from opencosmo.dataset.index import ChunkedIndex, DataIndex
from opencosmo.handler import OutOfMemoryHandler
from opencosmo.header import OpenCosmoHeader
from opencosmo.transformations import units as u

try:
    from mpi4py import MPI
except ImportError:
    MPI = None  # type: ignore


class DatasetBuilder(Protocol):
    """
    A DatasetBuilder is responsible for building a dataset from a file. It
    contains the logic for selecting columns and applying transformations to
    the data.
    """

    def with_units(self, convention: str) -> Self:
        pass

    def select(self, selected: str | Iterable[str]) -> Self:
        pass

    def build(
        self,
        file: File | Group,
        header: OpenCosmoHeader,
        index: Optional[DataIndex] = None,
    ) -> oc.Dataset:
        pass


class OomDatasetBuilder:
    __allowed_conventions = {
        "unitless",
        "scalefree",
        "comoving",
        "physical",
    }

    def __init__(
        self,
        selected: Optional[set[str]] = None,
        unit_convention: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.selected = selected
        self.unit_convention = (
            unit_convention if unit_convention is not None else "comoving"
        )

    def with_units(self, convention: str) -> OomDatasetBuilder:
        if convention not in self.__allowed_conventions:
            raise ValueError(
                f"Unit convention must be one of {self.__allowed_conventions}"
            )
        return OomDatasetBuilder(
            selected=self.selected,
            unit_convention=convention,
        )

    def select(self, selected: Iterable[str]) -> OomDatasetBuilder:
        selected = set(selected)
        if self.selected is None:
            return OomDatasetBuilder(
                selected=set(selected),
                unit_convention=self.unit_convention,
            )

        if not selected.issubset(self.selected):
            raise ValueError(
                "Selected columns must be a subset of the already selected columns."
            )
        return OomDatasetBuilder(
            selected=selected,
            unit_convention=self.unit_convention,
        )

    def build(
        self,
        file: File | Group,
        header: OpenCosmoHeader,
        index: Optional[DataIndex] = None,
    ) -> oc.Dataset:
        # tree = read_tree(file, header)
        tree = None
        builders, base_unit_transformations = u.get_default_unit_transformations(
            file, header
        )
        if self.selected is not None:
            selected = self.selected
        else:
            selected = builders.keys()

        if self.unit_convention != "comoving":
            new_transformations = u.get_unit_transition_transformations(
                self.unit_convention,
                base_unit_transformations,
                header.cosmology,
                header.file.redshift,
            )
            builders = get_column_builders(new_transformations, selected)

        if selected is not None:
            builders = {key: builders[key] for key in selected}

        handler = OutOfMemoryHandler(file, tree=tree)

        if index is None:
            index = ChunkedIndex.from_size(len(handler))

        dataset = oc.Dataset(
            handler,
            header,
            builders,
            base_unit_transformations,
            index,
        )
        return dataset
