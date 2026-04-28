from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Optional, Sequence
from weakref import finalize

import astropy.units as u
import numpy as np

from opencosmo.column.cache import ColumnCache
from opencosmo.column.column import RawColumn
from opencosmo.column.select import get_column_selection
from opencosmo.dataset.columns import add_columns, resort
from opencosmo.dataset.instantiate import instantiate_dataset
from opencosmo.dataset.output import get_derived_column_names, make_dataset_schema
from opencosmo.handler.empty import EmptyHandler
from opencosmo.handler.hdf5 import Hdf5Handler
from opencosmo.index.build import single_chunk
from opencosmo.index.mask import into_array
from opencosmo.index.unary import get_range
from opencosmo.plugins.plugin import PluginType, apply_index_plugins, apply_plugins
from opencosmo.units import UnitConvention
from opencosmo.units.handler import (
    make_unit_handler_from_hdf5,
    make_unit_handler_from_units,
)

if TYPE_CHECKING:
    from uuid import UUID

    from astropy import table
    from astropy.cosmology import Cosmology
    from numpy.typing import NDArray

    from opencosmo.column.column import ConstructedColumn
    from opencosmo.handler.protocols import DataCache, DataHandler
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.spatial.protocols import Region
    from opencosmo.units.handler import UnitHandler


def deregister_state(id: int, cache: DataCache):
    cache.deregister_column_group(id)


class DatasetState:
    """
    Holds mutable state required by the dataset. Cleans up the dataset to mostly focus
    on very high-level operations. Not a user facing class.
    """

    def __init__(
        self,
        column_producers: Sequence[ConstructedColumn],
        raw_data_handler: DataHandler,
        cache: DataCache,
        unit_handler: UnitHandler,
        header: OpenCosmoHeader,
        columns: dict[str, UUID],
        region: Region,
        sort_by: Optional[tuple[str, bool]],
    ):
        self.__producers: dict[UUID, ConstructedColumn] = {
            p.uuid: p for p in column_producers
        }
        self.__raw_data_handler = raw_data_handler
        self.__cache = cache
        self.__unit_handler = unit_handler
        self.__header = header
        self.__columns: dict[str, UUID] = columns
        self.__region = region
        self.__sort_by = sort_by
        self.__cache.register_column_group(id(self), self.__columns)
        finalize(self, deregister_state, id(self), self.__cache)

    def __rebuild(self, **updates):
        new = {
            "raw_data_handler": self.__raw_data_handler,
            "cache": self.__cache,
            "column_producers": list(self.__producers.values()),
            "unit_handler": self.__unit_handler,
            "header": self.__header,
            "columns": self.__columns,
            "region": self.__region,
            "sort_by": self.__sort_by,
        } | updates
        return DatasetState(**new)

    def __exit__(self, *exec_details):
        return None

    @classmethod
    def from_target(
        cls,
        target: DatasetTarget,
        unit_convention: UnitConvention,
        region: Region,
        index: Optional[DataIndex] = None,
        metadata_group: Optional[str] = None,
        in_memory: bool = False,
    ):
        data_group = target["dataset_group"]
        if "load" in data_group.keys():
            load_conditions = dict(data_group["load/if"].attrs)
        else:
            load_conditions = None

        handler = Hdf5Handler.from_columns(
            target["columns"],
            index,
            metadata_group,
            load_conditions,
        )
        unit_handler = make_unit_handler_from_hdf5(
            target["columns"], target["header"], unit_convention
        )
        descriptions = handler.descriptions

        producers = [
            RawColumn(cname, descriptions.get(cname, "None"))
            for cname in handler.columns
        ]
        columns = {p.name: p.uuid for p in producers}
        cache = ColumnCache.empty()
        return DatasetState(
            producers,
            handler,
            cache,
            unit_handler,
            target["header"],
            columns,
            region,
            None,
        )

    @classmethod
    def in_memory(
        cls,
        data_columns: dict,
        metadata_columns: dict,
        header: OpenCosmoHeader,
        unit_convention: UnitConvention,
        region: Region,
        descriptions: Optional[dict[str, str]] = None,
        index: Optional[DataIndex] = None,
    ):
        descriptions = descriptions or {}

        # Producers must be created first so their UUIDs are available for the cache.
        producers = [
            RawColumn(cname, descriptions.get(cname, "None"))
            for cname in data_columns.keys()
        ]
        columns = {p.name: p.uuid for p in producers}

        cache = ColumnCache.empty()
        if data_columns:
            uuid_data = {p.uuid: {p.name: data_columns[p.name]} for p in producers}
            cache.add_data(uuid_data, descriptions)
        if metadata_columns:
            cache.add_metadata(dict(metadata_columns), {})

        units: dict[str, u.Unit] = {}
        for name, column in data_columns.items():
            units[name] = None
            if isinstance(column, u.Quantity):
                units[name] = column.unit

        unit_handler = make_unit_handler_from_units(units, header, unit_convention)

        return DatasetState(
            producers,
            EmptyHandler(),
            cache,
            unit_handler,
            header,
            columns,
            region,
            None,
        )

    def __len__(self):
        if isinstance(self.__raw_data_handler, EmptyHandler):
            return len(self.__cache)
        return len(self.__raw_data_handler)

    @property
    def descriptions(self):
        all_descriptions = {}
        for producer in self.__producers.values():
            update = {name: producer.description for name in producer.produces}
            all_descriptions |= update
        all_descriptions |= self.__cache.descriptions

        return {
            name: description
            for name, description in all_descriptions.items()
            if name in self.columns
        }

    @property
    def raw_index(self):
        if (si := self.get_sorted_index()) is not None:
            ni = into_array(self.__raw_data_handler.index)
            return ni[si]

        return self.__raw_data_handler.index

    @property
    def unit_handler(self):
        return self.__unit_handler

    @property
    def units(self):
        units = self.__unit_handler.current_units
        return {name: units[name] for name in self.columns}

    @property
    def convention(self):
        return self.__unit_handler.current_convention

    @property
    def region(self):
        return self.__region

    @property
    def header(self):
        return self.__header

    @property
    def columns(self) -> list[str]:
        return list(self.__columns.keys())

    @property
    def meta_columns(self) -> list[str]:
        columns = set(self.__cache.metadata_columns).union(
            self.__raw_data_handler.metadata_columns
        )
        return list(columns)

    def get_data(
        self,
        ignore_sort: bool = False,
        metadata_columns: list = [],
        unit_kwargs: dict = {},
    ) -> table.QTable:
        """
        Get the data for a given handler.
        """
        state = apply_plugins(PluginType.DatasetInstantiate, self)

        data = instantiate_dataset(
            list(state.__producers.values()),
            state.__columns,
            state.__raw_data_handler,
            state.__cache,
            state.__unit_handler,
            unit_kwargs,
            metadata_columns,
            None if ignore_sort else state.__sort_by,
        )
        if missing := set(self.columns).difference(data.keys()):
            raise RuntimeError(
                f"Some columns are missing from the output! This is likely a bug. Please report it on GitHub. Missing: {missing}"
            )

        new_order = [c for c in self.columns]
        if metadata_columns:
            new_order.extend(metadata_columns)

        return {name: data[name] for name in new_order}

    def rows(self, metadata_columns: list = [], unit_kwargs: dict = {}):
        derived_to_collect = (
            set(self.columns)
            .difference(self.__cache.columns)
            .difference(self.__raw_data_handler.columns)
        )
        derived_storage: dict[str, list[np.ndarray]] = {
            name: [] for name in derived_to_collect
        }
        total_length = len(self)
        chunk_ranges = [
            (i, min(i + 1000, total_length)) for i in range(0, total_length, 1000)
        ]
        if not chunk_ranges:
            raise StopIteration

        try:
            for start, end in chunk_ranges:
                chunk = self.take_range(start, end)
                data = chunk.get_data(
                    metadata_columns=metadata_columns, unit_kwargs=unit_kwargs
                )
                for name in derived_to_collect:
                    derived_storage[name].append(data[name])

                for i in range(len(chunk)):
                    yield {name: column[i] for name, column in data.items()}
            all_derived = {
                name: np.concatenate(arr) for name, arr in derived_storage.items()
            }
            derived_storage = resort(all_derived, self.get_sorted_index())
            if derived_storage:
                uuid_keyed: dict = {}
                for name, arr in derived_storage.items():
                    uuid = self.__columns[name]
                    uuid_keyed.setdefault(uuid, {})[name] = arr
                self.__cache.add_data(uuid_keyed, {})
        except GeneratorExit:
            pass
        except BaseException:
            raise

    def get_metadata(self, columns=[]):
        metadata = self.__raw_data_handler.get_metadata(columns)
        sorted_index = self.get_sorted_index()
        if sorted_index is not None:
            metadata = {name: values[sorted_index] for name, values in metadata.items()}
        return metadata

    def with_mask(self, mask: NDArray[np.bool_]):
        return self.with_index(np.where(mask)[0])

    def with_index(self, index: DataIndex):
        index = apply_index_plugins(self, index)
        new_raw_handler = self.__raw_data_handler.take(index)
        new_cache = self.__cache.take(index)
        return self.__rebuild(
            cache=new_cache,
            raw_data_handler=new_raw_handler,
        )

    def make_schema(self, name: Optional[str] = None):
        producers = list(self.__producers.values())
        columns = set(self.__columns.keys())
        derived_names = get_derived_column_names(producers, columns)
        if derived_names:
            derived_data = (
                self.select(derived_names)
                .with_units(self.__unit_handler.base_convention, {}, {}, None, None)
                .get_data(ignore_sort=True)
            )
        else:
            derived_data = {}
        return make_dataset_schema(
            producers,
            self.__raw_data_handler,
            self.__cache,
            self.__columns,
            self.meta_columns,
            self.__header,
            self.__region,
            derived_data,
            name,
        )

    def with_new_columns(
        self,
        descriptions: dict[str, str] = {},
        allow_overwrite: bool = False,
        **new_columns: ConstructedColumn | np.ndarray | u.Quantity,
    ):
        """
        Add a set of derived columns to the dataset. A derived column is a column that
        has been created based on the values in another column.
        """
        new_producers, new_column_map, new_unit_handler = add_columns(
            list(self.__producers.values()),
            self.__unit_handler,
            self.__cache,
            self.__columns,
            self.get_sorted_index(),
            descriptions,
            new_columns,
            len(self),
            allow_overwrite=allow_overwrite,
        )
        return self.__rebuild(
            cache=self.__cache,
            column_producers=new_producers,
            columns=new_column_map,
            unit_handler=new_unit_handler,
        )

    def with_region(self, region: Region):
        """
        Return the same dataset but with a different region
        """
        return self.__rebuild(region=region)

    def select(self, columns: set[str], drop=False):
        """
        Select a subset of columns from the dataset. It is possible for a user to select
        a derived column in the dataset, but not the columns it is derived from.
        This class tracks any columns which are required to materialize the dataset but
        are not in the final selection in self.__hidden. When the dataset is
        materialized, the columns in self.__hidden are removed before the data is
        returned to the user.

        """

        selections, missing = get_column_selection(self.columns, columns)
        if missing:
            raise ValueError(
                f"Columns are included that are not in this dataset: {missing}"
            )
        elif not selections and columns:
            raise ValueError("No columns matched the provided wildcards!")

        if drop:
            selections = set(self.columns) - selections

        return self.__rebuild(columns={n: self.__columns[n] for n in selections})

    def sort_by(self, column_name: str, invert: bool):
        if column_name not in self.columns:
            raise ValueError(f"This dataset has no column {column_name}")

        return self.__rebuild(sort_by=(column_name, invert))

    def get_sorted_index(self):
        if self.__sort_by is not None:
            column = self.select({self.__sort_by[0]}).get_data(ignore_sort=True)[
                self.__sort_by[0]
            ]
            sorted = np.argsort(column)
            if self.__sort_by[1]:
                sorted = sorted[::-1]

        else:
            sorted = None

        return sorted

    def take(self, n: int, at: str):
        """
        Take rows from the dataset.
        """

        if at == "start":
            return self.take_range(0, n)
        elif at == "end":
            return self.take_range(len(self) - n, len(self))
        elif at == "random":
            row_indices = np.random.choice(len(self), n, replace=False)
            row_indices.sort()
        return self.take_rows(row_indices)

    def take_range(self, start: int, end: int):
        """
        Take a range of rows form the dataset.
        """
        if start < 0 or end < 0:
            raise ValueError("start and end must be positive.")
        if end < start:
            raise ValueError("end must be greater than start.")
        if end > len(self):
            raise ValueError("end must be less than the length of the dataset.")

        take_index = single_chunk(start, end - start)
        return self.take_rows(take_index)

    def take_rows(self, rows: DataIndex):
        if len(self) == 0:
            return self
        rows = apply_index_plugins(self, rows)
        row_range = get_range(rows)

        if row_range[1] > len(self) or row_range[0] < 0:
            raise ValueError(
                "Row indices must be between 0 and the length of this dataset!"
            )
        sorted = self.get_sorted_index()
        if sorted is not None:
            rows = np.sort(sorted[into_array(rows)])
        new_handler = self.__raw_data_handler.take(rows)
        new_cache = self.__cache.take(rows)

        return self.__rebuild(
            raw_data_handler=new_handler,
            cache=new_cache,
        )

    def with_units(
        self,
        convention: Optional[str],
        conversions: dict[u.Unit, u.Unit],
        columns: dict[str, u.Unit],
        cosmology: Cosmology,
        redshift: float | table.Column,
    ):
        """
        Change the unit convention
        """

        if convention is None:
            convention_ = self.__unit_handler.current_convention

        else:
            convention_ = UnitConvention(convention)

        if (
            convention_ == UnitConvention.SCALEFREE
            and UnitConvention(self.header.file.unit_convention)
            != UnitConvention.SCALEFREE
        ):
            raise ValueError(
                f"Cannot convert units with convention {self.header.file.unit_convention} to convention scalefree"
            )
        column_keys = set(columns.keys())
        missing_columns = column_keys - set(self.columns)
        if missing_columns:
            raise ValueError(f"Dataset does not have columns {missing_columns}")

        new_handler = self.__unit_handler.with_convention(convention_).with_conversions(
            conversions, columns
        )

        if convention_ == self.__unit_handler.current_convention:
            cache = self.__cache.create_child()
        else:
            all_derived_names: set[str] = set()
            all_derived_names = reduce(
                lambda acc, col: acc.union(
                    col.produces if not isinstance(col, RawColumn) else set()
                ),
                self.__producers.values(),
                all_derived_names,
            ).intersection(self.columns)
            columns_to_drop = all_derived_names.union(self.__raw_data_handler.columns)
            cache = self.__cache.drop(columns_to_drop)
        return self.__rebuild(unit_handler=new_handler, cache=cache)
