from __future__ import annotations

from typing import Optional

import numpy as np
from h5py import File, Group
from mpi4py import MPI

import opencosmo as oc
from opencosmo.dataset.column import ColumnBuilder
from opencosmo.handler import MPIHandler
from opencosmo.header import OpenCosmoHeader
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
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.selected = None
        self.file = file
        self.link = link
        self.header = header
        self.comm = comm
        self.tree = read_tree(file, header)
        self.builders, self.base_unit_transformations = (
            u.get_default_unit_transformations(file, header)
        )
        if isinstance(self.link, tuple):
            n_per_rank = self.link[0].shape[0] // self.comm.Get_size()
            self.offset = n_per_rank * self.comm.Get_rank()
        else:
            n_per_rank = self.link.shape[0] // self.comm.Get_size()
            self.offset = n_per_rank * self.comm.Get_rank()

    def get_all_data(self) -> oc.Dataset:
        return build_full_dataset(
            self.file,
            self.header,
            self.comm,
            self.tree,
            self.base_unit_transformations,
            self.builders,
        )

    def get_data(self, indices: int | np.ndarray) -> Optional[oc.Dataset]:
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
        dataset =  build_dataset(
            self.file,
            indices_into_data,
            self.header,
            self.comm,
            self.tree,
            self.base_unit_transformations,
            self.builders,
        )
        if self.selected is not None:
            dataset = dataset.select(self.selected)
        return dataset

    def select(self, columns: str | list[str]) -> MpiLinkHandler:
        if self.selected is not None:
            new_selected = set(columns)
            if not new_selected.issubset(self.selected):
                raise ValueError("Tried to select columns that are not in the dataset.")
        else:
            new_selected = set(columns)

        self.selected = new_selected

    def write(
        self, data_group: File, link_group: Group, name: str, indices: int | np.ndarray
    ):
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
            indices_into_data = self.link[self.offset + start:self.offset + end]
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
