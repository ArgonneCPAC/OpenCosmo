from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import h5py
    import numpy as np
    from astropy.table import Table
    from mpi4py import MPI

    from opencosmo import Dataset, Lightcone
    from opencosmo.dataset.state import DatasetState
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex, IndexArray
    from opencosmo.spatial.tree import Tree


class HookPoint(StrEnum):
    DatasetOpen = "dataset_open"
    DatasetInstantiate = "dataset_instantiate"
    LightconeOpen = "lightcone_open"
    LightconeInstantiate = "lightcone_instantiate"
    IndexUpdate = "index_update"
    PostSort = "post_sort"
    Partition = "partition"


# --- fold hooks ---
# Plugins receive and return these contexts. Use dataclasses.replace() to
# produce a modified copy rather than mutating in place.


@dataclass(frozen=True)
class DatasetOpenCtx:
    """Fired once per dataset after it is opened from disk.

    open_kwargs holds any keyword arguments the user passed to opencosmo.open().
    Plugins can inspect them to conditionally modify the dataset.
    """

    dataset: Dataset
    open_kwargs: dict[str, Any]


@dataclass(frozen=True)
class DatasetInstantiateCtx:
    """Fired each time get_data() is called on a DatasetState.

    Plugins may add or modify derived columns before the data is materialised.
    """

    state: DatasetState


@dataclass(frozen=True)
class LightconeOpenCtx:
    """Fired once per Lightcone after it is opened from disk.

    open_kwargs mirrors DatasetOpenCtx.open_kwargs.
    """

    lightcone: Lightcone
    open_kwargs: dict[str, Any]


@dataclass(frozen=True)
class LightconeInstantiateCtx:
    """Fired each time get_data() is called on a Lightcone.

    Plugins may re-order or re-index sub-datasets before they are stacked.
    """

    lightcone: Lightcone


@dataclass(frozen=True)
class IndexUpdateCtx:
    """Fired whenever a filter, take, or bound operation produces a new index.

    state is read-only context for the predicate and for reading column data.
    index is the value being transformed; return a modified copy via
    dataclasses.replace(ctx, index=new_index).
    """

    state: DatasetState
    index: DataIndex


@dataclass(frozen=True)
class PostSortCtx:
    """Fired after a sort operation reorders rows.

    state is read-only context for the predicate.
    index is the reverse-sort permutation (i.e. np.argsort of the sort order),
    which can be used to remap index-valued columns.
    data is the value being transformed; return a modified copy via
    dataclasses.replace(ctx, data=new_data).
    """

    state: DatasetState | Lightcone
    data: Table | dict[str, np.ndarray]
    index: IndexArray


# --- query hook ---
# Partition uses query() rather than fold(). The plugin returns a
# TreePartition directly (not a modified context), and the first non-None
# result wins. No plugin responding means the caller falls back to the
# default partitioning strategy.


@dataclass(frozen=True)
class PartitionCtx:
    """Fired during MPI open to determine how rows are distributed across ranks.

    Plugins return a TreePartition, or None to defer to the default strategy.
    This hook uses query() semantics: at most one plugin responds.
    """

    comm: MPI.Comm
    header: OpenCosmoHeader
    index_group: h5py.Group
    data_group: h5py.Group
    tree: Optional[Tree] = None
    min_level: Optional[int] = None
