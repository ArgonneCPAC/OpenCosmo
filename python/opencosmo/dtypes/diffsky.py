from __future__ import annotations

from datetime import datetime  # noqa
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer

from opencosmo.column.column import EvaluatedColumn, EvaluateStrategy
from opencosmo.index import into_array
from opencosmo.index.ops import reindex_column
from opencosmo.mpi import get_mpi
from opencosmo.plugins.plugin import (
    IndexPluginSpec,
    PartitionPluginSpec,
    PluginSpec,
    PluginType,
    PostSortPluginSpec,
    register_plugin,
)
from opencosmo.spatial.tree import TreePartition

if TYPE_CHECKING:
    import h5py
    from astropy.table import Table
    from mpi4py import MPI

    from opencosmo import Dataset, Lightcone
    from opencosmo.dataset.state import DatasetState
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.spatial.tree import Tree


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


def top_host_idx_plugin(dataset: DatasetState, **kwargs):
    top_host_idx = EvaluatedColumn(
        rebuild_top_host_idx,
        requires=set(["top_host_idx"]),
        produces=set(["top_host_idx"]),
        format="numpy",
        units={"top_host_idx": None},
        strategy=EvaluateStrategy.VECTORIZE,
        no_cache=True,
    )
    return dataset.with_new_columns(updated_host_idx=top_host_idx, allow_overwrite=True)


def top_host_idx_offset_plugin(lightcone: Lightcone) -> dict[str, Dataset]:
    cs = 0
    output = {}

    def top_host_idx(top_host_idx, offset):
        top_host_idx[top_host_idx >= 0] += offset
        return top_host_idx

    for key, ds in lightcone.items():
        output[key] = ds.evaluate(
            top_host_idx, allow_overwrite=True, vectorize=True, offset=cs
        )
        cs += len(ds)

    return output


def top_host_idx_verifier[T: (DatasetState, Dataset, Lightcone)](
    dataset: T, **kwargs
) -> bool:
    return (
        dataset.header.file.data_type == "synthetic_galaxies"
        and "top_host_idx" in dataset.columns
    )


def keep_top_host_idx(dataset: DatasetState, new_index: DataIndex):
    index_array = into_array(new_index)
    top_host_idx = dataset.select({"top_host_idx"}).get_data()["top_host_idx"]
    unique_in_sample = np.unique(top_host_idx[index_array])

    missing_hosts = np.setdiff1d(unique_in_sample, index_array)
    all_satellites = np.where(np.isin(top_host_idx, unique_in_sample))[0]
    missing_satellites = np.setdiff1d(all_satellites, index_array)

    if len(missing_hosts) == 0 and len(missing_satellites) == 0:
        return new_index

    all_missing = np.sort(np.concatenate((missing_hosts, missing_satellites)))

    insert_idx = np.searchsorted(index_array, all_missing)
    result = np.insert(index_array, insert_idx, all_missing)

    return result


def keep_top_host_idx_verifier(dataset: DatasetState):
    return "top_host_idx" in dataset.columns


def update_top_host_idx_after_sort(data: Table, reverse_index: DataIndex):
    mask = data["top_host_idx"] >= 0
    data["top_host_idx"][mask] = reverse_index[data["top_host_idx"][mask]]
    return data


def register_keep_top_host_idx(dataset: Dataset, **kwargs):
    register_plugin(
        IndexPluginSpec(
            PluginType.IndexUpdate, keep_top_host_idx_verifier, keep_top_host_idx
        )
    )
    return dataset


def register_keep_top_host_idx_verifier(
    dataset: Dataset, keep_top_host: bool = False, **kwargs
):
    return keep_top_host and top_host_idx_verifier(dataset)


def partition_plugin(
    comm: MPI.Comm,
    index_group: h5py.Group,
    data_group: h5py.Group,
    tree: Optional[Tree] = None,
    min_level: Optional[int] = None,
):
    top_host_idx = data_group["top_host_idx"][:]
    if comm.Get_rank() == 0:
        n_ranks = comm.Get_size()
        unique_hosts = np.unique(top_host_idx, sorted=True)
        ave, res = divmod(unique_hosts.size, n_ranks)
        counts = np.array(
            [ave + 1 if i < res else ave for i in range(n_ranks)], dtype=np.int64
        )
        displs = np.cumsum(np.concatenate(([0], counts[:-1])))
        counts = comm.bcast(counts)
        # Should always be roughly balanced, since data is
        # ordered spatially
        rank_top_hosts = np.zeros(counts[0], dtype=np.int64)
        comm.Scatterv([unique_hosts, counts, displs, MPI.INT64_T], rank_top_hosts)
    else:
        counts = comm.bcast(None)
        rank_top_hosts = np.zeros(counts[comm.Get_rank()], dtype=np.int64)
        comm.Scatterv([None, counts, None, MPI.INT64_T], rank_top_hosts)

    is_in_groups = np.isin(top_host_idx, rank_top_hosts)
    index = np.where(is_in_groups)[0]
    return TreePartition(idx=index, region=None, level=None)


def partition_plugin_verifier(
    header: OpenCosmoHeader,
    index_group: h5py.Group,
    data_group: h5py.Group,
    tree: Optional[Tree] = None,
    min_level: Optional[int] = None,
):
    return (
        header.file.data_type == "synthetic_galaxies"
        and "top_host_idx" in data_group.keys()
    )


register_plugin(
    PluginSpec(PluginType.DatasetOpen, top_host_idx_verifier, top_host_idx_plugin)
)

register_plugin(
    PluginSpec(
        PluginType.LightconeInstantiate,
        top_host_idx_verifier,
        top_host_idx_offset_plugin,
    )
)

register_plugin(
    PluginSpec(
        PluginType.LightconeOpen,
        register_keep_top_host_idx_verifier,
        register_keep_top_host_idx,
    )
)

register_plugin(
    PostSortPluginSpec(
        PluginType.PostSort, top_host_idx_verifier, update_top_host_idx_after_sort
    )
)

register_plugin(
    PartitionPluginSpec(
        PluginType.Partition, partition_plugin_verifier, partition_plugin
    )
)
