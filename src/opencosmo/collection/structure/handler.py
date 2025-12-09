from __future__ import annotations

from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np

from opencosmo.index import into_array

if TYPE_CHECKING:
    import opencosmo as oc

"""
A tale in 3 acts:

Act 1: Patrick creates a LinkHandler which holds a pointer to the index that determines
which rows in halo/galaxy properties corresponds to rows in particle datasets.

Act 2: Patrick unifies metadata handling in Datasets, making the link handler
unncessary. Instead, ephemeral particle datasets are created when requested.

Act 3: Patrick realizes this solution makes it impossible to cache things, particularly
very expensive computations that the user has created with evaluate. Patrick re-introduces
the LinkHandler, but it's better this time or something.

"""

LINK_ALIASES = {  # Left: Name in file, right: Name in collection
    "sodbighaloparticles_star_particles": "star_particles",
    "sodbighaloparticles_dm_particles": "dm_particles",
    "sodbighaloparticles_gravity_particles": "gravity_particles",
    "sodbighaloparticles_agn_particles": "agn_particles",
    "sodbighaloparticles_gas_particles": "gas_particles",
    "sod_profile": "halo_profiles",
    "galaxyproperties": "galaxy_properties",
    "galaxyparticles_star_particles": "star_particles",
}


def create_start_size(data, start_name, size_name):
    start = data.pop(start_name, None)
    size = data.pop(size_name, None)
    if start is None:
        return None
    valid = size > 0
    if isinstance(start, np.ndarray):
        return (start[valid], size[valid])
    if size == 0:
        return None
    return (np.atleast_1d(start), np.atleast_1d(size))


def create_idx(data, idx_name):
    idx = data.pop(idx_name, None)
    if idx is None:
        return None

    valid = idx >= 0

    if isinstance(idx, np.ndarray):
        return idx[valid]
    elif idx == -1:
        return None
    return np.atleast_1d(idx)


def make_links(keys, rename_galaxies=False):
    starts = list(filter(lambda key: "start" in key, keys))
    sizes = list(filter(lambda key: "size" in key, keys))
    idxs = list(filter(lambda key: "idx" in key, keys))

    starts = set(map(lambda key: key[:-6], starts))
    sizes = set(map(lambda key: key[:-5], sizes))
    idxs = set(map(lambda key: key[:-4], idxs))

    assert starts == sizes
    output = {}
    columns = {}
    for name in starts:
        output[LINK_ALIASES[name]] = partial(
            create_start_size, start_name=f"{name}_start", size_name=f"{name}_size"
        )
        columns[LINK_ALIASES[name]] = [f"{name}_start", f"{name}_size"]

    for name in idxs:
        output[LINK_ALIASES[name]] = partial(create_idx, idx_name=f"{name}_idx")
        columns[LINK_ALIASES[name]] = [f"{name}_idx"]

    if rename_galaxies and "galaxy_properties" in output:
        output["galaxies"] = output.pop("galaxy_properties")
        columns["galaxies"] = columns.pop("galaxy_properties")
    return output, columns


class LinkHandler:
    """
    This needs some explanation. We break the "don't mutate state" rule pretty hard here.

    When a StructureCollection is initialized, we build its linked datasets based on
    the metadata in the halo/galaxy properties which tells us which rows in the linked
    datasets belong to which halo/galaxy.

    When a StructureCollection is modified (e.g. by filtering) we filter the source
    dataset but DO NOT update the linked datasets, because doing so is a fairly
    expensive operation. Instead, we create a LinkHandler that knows which rows in
    the original halo/galaxy properties are included in the current version of the
    datasets.

    When data is requested, we perform the update if necessary. The StructureCollection
    passes in its current halo/galaxy properties and the un-updated linked datasets. We know
    for sure that the rows in the current halo/galaxy properties is a strict subset of the rows
    in the original halo/galaxy properties. We determine where the overlaps are, and then update
    the datasets accordingly.

    The original version of the StructureCollection held the entire, unmodified linked datasets
    and created the version with the correct rows when the data was requested. While this was very clean,
    it made it impossible to cache data because the linked datasets that actually got returned to the
    user were ephemeral.

    We could re-build the datasets each time we create a new StructureCollection, but this is quite slow.
    Converting from that approach to this approach shaved 25% off the runtime of one of my tests.
    """

    def __init__(
        self,
        links,
        columns,
        derived_from: Optional[oc.Dataset],
    ):
        self.__derived_from = derived_from
        self.links = links
        self.columns = columns

    @classmethod
    def from_link_names(cls, names: Iterable[str], rename_galaxies=False):
        links, columns = make_links(names, rename_galaxies)
        return LinkHandler(links, columns, None)

    def parse(self, data: dict[str, Any]):
        output = {}
        for name, handler in self.links.items():
            result = handler(data)
            if result is not None:
                output[name] = result
        return output

    def prep_datasets(self, source: oc.Dataset, datasets: dict[str, oc.Dataset]):
        """
        Called once when a datasets are opened for the first time. Downstream
        versions always use rebuild_datsets
        """

        all_columns: list[str] = reduce(
            lambda acc, ds: acc + self.columns[ds], datasets.keys(), []
        )
        meta = source.get_metadata(all_columns)
        indices = self.parse(meta)
        new_datasets = datasets
        for name, index in indices.items():
            new_datasets[name] = new_datasets[name].take_rows(index)
        return new_datasets

    def make_derived(self, source: oc.Dataset):
        """
        Because the library encourages performing several operations in a row before requesting data,
        it is a bad idea to re-build the datasets every time someone performs an operation. Instead,
        we just hold a copy of the source dataset the last time the datasets were rebuilt,
        which allows us to perform the rebuild whenver data is actually requsted.
        """
        derived_from = self.__derived_from
        if self.__derived_from is None:
            derived_from = source

        return LinkHandler(self.links, self.columns, derived_from)

    def rebuild_datasets(
        self,
        new_source: oc.Dataset,
        datasets: dict[str, oc.Dataset],
    ):
        """
        We have a few guarantees here:
        1. The rows in new_source is a strict subset of the rows in source
        2. The rows in both are unique

        What is NOT guaranteed:
        1. The rows are sorted
        """
        if self.__derived_from is None:
            return datasets
        original_index = into_array(self.__derived_from.index)
        new_index = into_array(new_source.index)

        _, index_into_original, index_into_new = np.intersect1d(
            original_index, new_index, assume_unique=True, return_indices=True
        )
        all_columns: list[str] = reduce(
            lambda acc, ds: acc + self.columns[ds], datasets.keys(), []
        )
        index_into_original = index_into_original[np.argsort(index_into_new)]
        metadata = self.__derived_from.get_metadata(all_columns)
        new_datasets = {}

        for name, dataset in datasets.items():
            if len(self.columns[name]) == 1:
                metadata_column = metadata[self.columns[name][0]]
                new_datasets[name] = rebuild_row_index(
                    new_source, dataset, metadata_column, index_into_original
                )
            else:
                size_column = [name for name in self.columns[name] if "size" in name]
                assert len(size_column) == 1
                size_column_name = size_column[0]
                size_column_data = metadata[size_column_name]
                new_datasets[name] = rebuild_chunk_index(
                    new_source, dataset, size_column_data, index_into_original
                )
        return new_datasets

    def resort(self, source: oc.Dataset, datasets: dict[str, oc.Dataset]):
        """
        Data is always written in its original order, whether or not it has been sorted.
        This is to preserve the spatial index. However, when linked datasets are rebuilt
        they are rebuilt in the sorted order. This method re-sorts them based on the
        index from the original data.
        """
        all_columns: list[str] = reduce(
            lambda acc, ds: acc + self.columns[ds], datasets.keys(), []
        )
        all_columns = list(
            filter(lambda name: "idx" in name or "size" in name, all_columns)
        )

        sort_index = np.argsort(into_array(source.index))

        if np.all(sort_index[1:] >= sort_index[:-1]):
            # Already sorted. Carry on!
            return datasets

        meta = source.get_metadata(all_columns)
        output = {}
        for name, dataset in datasets.items():
            if len(self.columns[name]) == 1:
                valid_rows = meta[self.columns[name][0]] >= 0
                new_dataset = dataset.take_rows(sort_index[valid_rows])
            else:
                size_column = [name for name in self.columns[name] if "size" in name]
                assert len(size_column) == 1
                size_column_data = meta[size_column[0]]
                chunk_boundaries = np.zeros(len(size_column_data) + 1, dtype=int)
                _ = np.cumsum(size_column_data, out=chunk_boundaries[1:])
                starts = chunk_boundaries[sort_index]
                sizes = size_column_data[sort_index]
                valid = sizes > 0
                idx = (starts[valid], sizes[valid])
                new_dataset = dataset.take_rows(idx)
            output[name] = new_dataset
        return output


def rebuild_row_index(
    new_source: oc.Dataset,
    dataset: oc.Dataset,
    original_metadata_column: np.ndarray,
    index_into_original: np.ndarray,
):
    valid_rows = original_metadata_column >= 0
    index = np.full(len(original_metadata_column), -1, dtype=int)
    index[valid_rows] = np.arange(0, sum(valid_rows))
    index_to_take = index[index_into_original]
    index_to_take = index_to_take[index_to_take >= 0]

    return dataset.take_rows(index_to_take)


def rebuild_chunk_index(
    new_source: oc.Dataset,
    dataset: oc.Dataset,
    original_size_column: np.ndarray,
    index_into_original: np.ndarray,
):
    chunk_boundaries = np.zeros(len(original_size_column) + 1, dtype=int)
    _ = np.cumsum(original_size_column, out=chunk_boundaries[1:])
    valid_rows = original_size_column[index_into_original] > 0

    starts = chunk_boundaries[index_into_original[valid_rows]]
    sizes = original_size_column[index_into_original[valid_rows]]

    ds = dataset.take_rows((starts, sizes))
    return ds
