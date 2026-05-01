from __future__ import annotations

import dataclasses
from datetime import datetime  # noqa
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer

import opencosmo.dataset.state as st
from opencosmo.column.column import EvaluatedColumn, EvaluateStrategy
from opencosmo.index import into_array
from opencosmo.index.ops import reindex_column
from opencosmo.mpi import get_mpi
from opencosmo.plugins.contexts import HookPoint
from opencosmo.plugins.hook import hook
from opencosmo.spatial.tree import TreePartition

if TYPE_CHECKING:
    from astropy.table import Table
    from mpi4py import MPI

    from opencosmo import Dataset
    from opencosmo.dataset.state import DatasetState
    from opencosmo.index import DataIndex
    from opencosmo.plugins.contexts import (
        DatasetOpenCtx,
        IndexUpdateCtx,
        LightconeInstantiateCtx,
        PartitionCtx,
        PostSortCtx,
    )

else:
    MPI = get_mpi()


class DiffskyVersionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "diffsky_versions"
    diffmah: str
    diffsky: str
    diffstar: str
    diffstarpop: Optional[str] = None
    dsps: str
    jax: str
    numpy: str


class DiffskyCatalogInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "catalog_info"
    README: Optional[str] = None
    mock_version_name: str
    zphot_table: Optional[tuple[float, ...]] = None

    @field_serializer("zphot_table")
    def serialize_zphot_table(self, value):
        if value is not None:
            return list(value)
        return None


# --- pure logic ---


def __offset(top_host_idx, offset):
    output = top_host_idx
    output[output >= 0] += offset
    return {"top_host_idx": output}


def offset_top_host_idx(datasets: list[Dataset]):
    lengths = [len(ds) for ds in datasets]
    offsets = np.cumsum(lengths)
    output_datasets = [datasets[0]]
    for offset, ds in zip(offsets, datasets[1:]):
        output_ds = ds.evaluate(
            __offset,
            offset=offset,
            vectorize=True,
            allow_overwrite=True,
        )
        output_datasets.append(output_ds)  # type: ignore
    return output_datasets


def rebuild_top_host_idx(top_host_idx, index):
    result = reindex_column(index, top_host_idx)
    return {"top_host_idx": result}


def keep_top_host_idx(dataset: DatasetState, new_index: DataIndex):
    index_array = into_array(new_index)
    top_host_idx = st.get_data(st.select(dataset, {"top_host_idx"}))["top_host_idx"]
    unique_in_sample = np.unique(top_host_idx[index_array])

    missing_hosts = np.setdiff1d(unique_in_sample, index_array)
    all_satellites = np.where(np.isin(top_host_idx, unique_in_sample))[0]
    missing_satellites = np.setdiff1d(all_satellites, index_array)

    if len(missing_hosts) == 0 and len(missing_satellites) == 0:
        return new_index

    all_missing = np.sort(np.concatenate((missing_hosts, missing_satellites)))
    insert_idx = np.searchsorted(index_array, all_missing)
    return np.insert(index_array, insert_idx, all_missing)


def _is_synthetic_galaxies_with_top_host_idx(obj) -> bool:
    return (
        obj.header.file.data_type == "synthetic_galaxies"
        and "top_host_idx" in obj.columns
    )


# --- hooks ---


@hook(
    HookPoint.DatasetOpen,
    when=lambda ctx: _is_synthetic_galaxies_with_top_host_idx(ctx.dataset),
)
def _attach_top_host_idx_column(ctx: DatasetOpenCtx) -> DatasetOpenCtx:
    top_host_idx = EvaluatedColumn(
        rebuild_top_host_idx,
        requires={"top_host_idx"},
        produces={"top_host_idx"},
        format="numpy",
        units={"top_host_idx": None},
        strategy=EvaluateStrategy.VECTORIZE,
        no_cache=True,
    )
    new_dataset = ctx.dataset.with_new_columns(
        updated_host_idx=top_host_idx, allow_overwrite=True
    )
    return dataclasses.replace(ctx, dataset=new_dataset)


@hook(
    HookPoint.LightconeInstantiate,
    when=lambda ctx: _is_synthetic_galaxies_with_top_host_idx(ctx.lightcone),
)
def _offset_top_host_idx(ctx: LightconeInstantiateCtx) -> LightconeInstantiateCtx:
    cs = 0
    output = {}

    def _offset_top_host_idx(top_host_idx, offset):
        top_host_idx[top_host_idx >= 0] += offset
        return top_host_idx

    for key, ds in ctx.lightcone.items():
        output[key] = ds.evaluate(
            _offset_top_host_idx, allow_overwrite=True, vectorize=True, offset=cs
        )
        cs += len(ds)

    return dataclasses.replace(ctx, lightcone=output)  # type: ignore[arg-type]


# Registers an IndexUpdate hook dynamically so that keep_top_host_idx only
# activates when the user explicitly requests it via open(..., keep_top_host=True).
@hook(
    HookPoint.IndexUpdate,
    when=lambda ctx: (
        "top_host_idx" in ctx.state.columns
        and ctx.state.kwargs.get("keep_top_host", False)
    ),
)
def _keep(ctx: IndexUpdateCtx) -> IndexUpdateCtx:
    return dataclasses.replace(ctx, index=keep_top_host_idx(ctx.state, ctx.index))


@hook(
    HookPoint.PostSort,
    when=lambda ctx: _is_synthetic_galaxies_with_top_host_idx(ctx.state),
)
def _remap_top_host_idx_after_sort(ctx: PostSortCtx) -> PostSortCtx:
    data: Table = ctx.data  # type: ignore[assignment]
    mask = data["top_host_idx"] >= 0
    data["top_host_idx"][mask] = ctx.index[data["top_host_idx"][mask]]
    return ctx


@hook(
    HookPoint.Partition,
    when=lambda ctx: (
        ctx.header.file.data_type == "synthetic_galaxies"
        and "top_host_idx" in ctx.data_group.keys()
    ),
)
def _partition_by_top_host_groups(ctx: PartitionCtx) -> Optional[TreePartition]:
    top_host_idx = ctx.data_group["top_host_idx"][:]
    n_rows = len(top_host_idx)
    n_ranks = ctx.comm.Get_size()
    rank = ctx.comm.Get_rank()

    ave, res = divmod(n_rows, n_ranks)
    counts = np.array(
        [ave + 1 if i < res else ave for i in range(n_ranks)], dtype=np.int64
    )
    displs = np.cumsum(np.concatenate(([0], counts[:-1])))
    start = displs[rank]
    count = counts[rank]
    row_indices = np.arange(start, start + count, dtype=np.int64)
    chunk = top_host_idx[start : start + count]

    # find top hosts (self-referential) and orphans (top_host_idx == -1)
    rank_top_hosts = row_indices[chunk == row_indices]
    rank_orphans = row_indices[chunk == -1]

    # gather all rows belonging to this rank's top hosts
    all_group_rows = np.where(np.isin(top_host_idx, rank_top_hosts))[0]

    index = np.union1d(all_group_rows, rank_orphans)
    return TreePartition(idx=index, region=None, level=None)
