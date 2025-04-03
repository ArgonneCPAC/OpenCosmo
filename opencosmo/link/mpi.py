from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from h5py import File, Group
from mpi4py import MPI

import opencosmo as oc
from opencosmo.dataset.column import ColumnBuilder, get_column_builders
from opencosmo.handler import MPIHandler
from opencosmo.header import OpenCosmoHeader
from opencosmo.link.builder import DatasetBuilder
from opencosmo.spatial import Tree, read_tree
from opencosmo.transformations import TransformationDict
from opencosmo.transformations import units as u


def build_dataset(
    file: File | Group,
    indices: np.ndarray,
    header: OpenCosmoHeader,
    comm: MPI.Comm,
    tree: Tree,
    base_transformations: TransformationDict,
    builders: dict[str, ColumnBuilder],
) -> oc.Dataset:
    if len(indices) > 0:
        index_range = (indices.min(), indices.max() + 1)
        indices = indices - index_range[0]
    else:
        index_range = None

    handler = MPIHandler(file, tree=tree, comm=comm, rank_range=index_range)
    return oc.Dataset(handler, header, builders, base_transformations, indices)


def build_full_dataset(
    file: File | Group,
    header: OpenCosmoHeader,
    comm: MPI.Comm,
    tree: Tree,
    base_transformations: TransformationDict,
    builders: dict[str, ColumnBuilder],
) -> oc.Dataset:
    handler = MPIHandler(file, tree=tree, comm=comm)
    return oc.Dataset(
        handler, header, builders, base_transformations, np.arange(len(handler))
    )


class MpiLinkHandler:
    def __init__(
        self,
        file: File | Group,
        link: Group | tuple[Group, Group],
        header: OpenCosmoHeader,
        builder: Optional[DatasetBuilder] = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        **kwargs,
    ):
        self.selected: Optional[set[str]] = None
        self.file = file
        self.link = link
        self.header = header
        self.comm = comm
        if builder is None:
            tree = read_tree(file, self.header)
            builder = MpiDatasetBuilder(tree, comm=comm)
        self.builder = builder
        if isinstance(self.link, tuple):
            n_per_rank = self.link[0].shape[0] // self.comm.Get_size()
            self.offset = n_per_rank * self.comm.Get_rank()
        else:
            n_per_rank = self.link.shape[0] // self.comm.Get_size()
            self.offset = n_per_rank * self.comm.Get_rank()

    def get_all_data(self) -> oc.Dataset:
        return self.builder.build(
            self.file,
            self.header,
        )

    def get_data(self, indices: int | np.ndarray) -> oc.Dataset:
        if isinstance(indices, int):
            indices = np.array([indices], dtype=int)

        if isinstance(self.link, tuple):
            start = self.link[0][indices + self.offset]
            size = self.link[1][indices + self.offset]
            valid_rows = size > 0
            start = start[valid_rows]
            size = size[valid_rows]
            if len(start) == 0:
                indices_into_data = np.array([], dtype=int)
            else:
                indices_into_data = np.concatenate(
                    [np.arange(idx, idx + length) for idx, length in zip(start, size)]
                )
        else:
            indices_into_data = self.link[indices + self.offset]
            indices_into_data = indices_into_data[indices_into_data >= 0]
            if len(indices_into_data) == 0:
                indices_into_data = np.array([], dtype=int)

        dataset = self.builder.build(
            self.file,
            self.header,
            indices=indices_into_data,
        )

        return dataset

    def with_units(self, convention: str) -> MpiLinkHandler:
        new_builder = self.builder.with_units(convention)
        return MpiLinkHandler(
            self.file,
            self.link,
            self.header,
            comm=self.comm,
            builder=new_builder,
        )

    def select(self, columns: str | Iterable[str]) -> MpiLinkHandler:
        new_builder = self.builder.select(columns)
        return MpiLinkHandler(
            self.file,
            self.link,
            self.header,
            comm=self.comm,
            builder=new_builder,
        )

    def write(
        self, data_group: Group, link_group: Group, name: str, indices: int | np.ndarray
    ) -> None:
        # Pack the indices
        if isinstance(indices, int):
            indices = np.array([indices])
        sizes = self.comm.allgather(len(indices))
        shape = (sum(sizes),)
        if sum(sizes) == 0:
            return

        if not isinstance(self.link, tuple):
            link_group.create_dataset("sod_profile_idx", shape=shape, dtype=int)
            self.comm.Barrier()
            start = indices[0]
            end = indices[-1] + 1
            indices_into_data = self.link[self.offset + start : self.offset + end]
            indices_into_data = indices_into_data[indices - start]
            nonzero = indices_into_data >= 0
            nonzero = self.comm.gather(nonzero)

            if self.comm.Get_rank() == 0:
                nonzero = np.concatenate(nonzero)
                sod_profile_idx = np.full(len(nonzero), -1)
                sod_profile_idx[nonzero] = np.arange(sum(nonzero))
                link_group["sod_profile_idx"][:] = sod_profile_idx
        else:
            link_group.create_dataset(f"{name}_start", shape=shape, dtype=int)
            link_group.create_dataset(f"{name}_size", shape=shape, dtype=int)
            self.comm.Barrier()
            rank_sizes = self.link[1][self.offset + indices]
            all_rank_sizes = self.comm.gather(rank_sizes)
            if self.comm.Get_rank() == 0:
                if all_rank_sizes is None:
                    # should never happen, but mypy...
                    raise ValueError("No data to write")

                all_sizes = np.concatenate(all_rank_sizes)
                starts = np.insert(np.cumsum(all_sizes), 0, 0)[:-1]
                link_group[f"{name}_start"][:] = starts
                link_group[f"{name}_size"][:] = all_sizes

        dataset = self.get_data(indices)

        if dataset is not None:
            dataset.write(data_group, name)


class MpiDatasetBuilder:
    __allowed_conventions = {
        "unitless",
        "scalefree",
        "comoving",
        "physical",
    }

    def __init__(
        self,
        tree: Tree,
        selected: Optional[set[str]] = None,
        unit_convention: Optional[str] = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.tree = tree
        self.selected = selected
        self.unit_convention = (
            unit_convention if unit_convention is not None else "comoving"
        )
        self.comm = comm

    def with_units(self, convention: str) -> MpiDatasetBuilder:
        if convention not in self.__allowed_conventions:
            raise ValueError(
                f"Unit convention must be one of {self.__allowed_conventions}"
            )
        return MpiDatasetBuilder(
            tree=self.tree,
            selected=self.selected,
            unit_convention=convention,
            comm=self.comm,
        )

    def select(self, selected: str | Iterable[str]) -> MpiDatasetBuilder:
        if isinstance(selected, str):
            selected = [selected]
        selected = set(selected)
        if self.selected is None:
            return MpiDatasetBuilder(
                tree=self.tree,
                selected=selected,
                unit_convention=self.unit_convention,
                comm=self.comm,
            )
        if not selected.issubset(self.selected):
            raise ValueError(
                "Selected columns must be a subset of the already selected columns."
            )
        return MpiDatasetBuilder(
            tree=self.tree,
            selected=self.selected.intersection(selected),
            unit_convention=self.unit_convention,
            comm=self.comm,
        )

    def build(
        self,
        file: File | Group,
        header: OpenCosmoHeader,
        indices: Optional[np.ndarray] = None,
    ) -> oc.Dataset:
        builders, base_unit_transformations = u.get_default_unit_transformations(
            file, header
        )
        selected = self.selected
        if selected is None:
            selected = builders.keys()

        if self.unit_convention != "comoving":
            new_transformations = u.get_unit_transition_transformations(
                self.unit_convention, base_unit_transformations, header.cosmology
            )
            builders = get_column_builders(new_transformations, selected)

        builders = {key: builders[key] for key in selected}

        rank_range = None
        if indices is not None and len(indices) > 0:
            rank_range = (indices.min(), indices.max() + 1)
            indices = indices - rank_range[0]

        handler = MPIHandler(
            file, tree=self.tree, comm=self.comm, rank_range=rank_range
        )
        if indices is None:
            indices = np.arange(len(handler))

        dataset = oc.Dataset(
            handler,
            header,
            builders,
            base_unit_transformations,
            indices,
        )

        return dataset
