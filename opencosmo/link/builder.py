from __future__ import annotations

from typing import Iterable, Optional, Protocol, Self

import numpy as np
from h5py import File, Group

from opencosmo import Dataset
from opencosmo.dataset.column import get_column_builders
from opencosmo.handler import OutOfMemoryHandler
from opencosmo.header import OpenCosmoHeader
from opencosmo.spatial import read_tree
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

    def __init__(
        self,
        selected: Optional[set[str]] = None,
        unit_convention: Optional[str] = None,
        *args,
        **kwargs,
    ):
        pass

    def with_units(self, convention: str) -> Self:
        pass

    def select(self, selected: Iterable[str]) -> Self:
        pass

    def build(
        self,
        file: File | Group,
        header: OpenCosmoHeader,
        indices: Optional[np.ndarray] = None,
    ) -> Dataset:
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
        indices: Optional[np.ndarray] = None,
    ) -> Dataset:
        tree = read_tree(file, header)
        builders, base_unit_transformations = u.get_default_unit_transformations(
            file, header
        )
        if self.selected is not None:
            selected = self.selected
        else:
            selected = builders.keys()

        if self.unit_convention != "comoving":
            new_transformations = u.get_unit_transition_transformations(
                self.unit_convention, base_unit_transformations, header.cosmology
            )
            builders = get_column_builders(new_transformations, selected)

        if selected is not None:
            builders = {key: builders[key] for key in selected}

        handler = OutOfMemoryHandler(file, tree=tree)

        if indices is None:
            indices_ = np.arange(len(handler))

        elif len(indices) > 0:
            if indices[0] < 0 or indices[-1] >= len(handler):
                raise ValueError(
                    "Indices must be within 0 and the length of the dataset."
                )
            indices_ = indices
        else:
            indices_ = indices

        dataset = Dataset(
            handler,
            header,
            builders,
            base_unit_transformations,
            indices_,
        )
        return dataset
