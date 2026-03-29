from __future__ import annotations

from copy import copy
from functools import reduce
from typing import TYPE_CHECKING, Iterable, Optional
from weakref import finalize

import astropy.units as u
import numpy as np

from opencosmo.column.cache import ColumnCache
from opencosmo.column.column import Column, DerivedColumn, EvaluatedColumn
from opencosmo.column.select import get_column_selection
from opencosmo.dataset.derived import (
    build_derived_columns,
    validate_derived_columns,
)
from opencosmo.dataset.im import resort, validate_in_memory_columns
from opencosmo.handler.empty import EmptyHandler
from opencosmo.handler.hdf5 import Hdf5Handler
from opencosmo.index.build import single_chunk
from opencosmo.index.mask import into_array
from opencosmo.index.unary import get_range
from opencosmo.io.schema import FileEntry, combine_with_cached_schema, make_schema
from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter, NumpySource
from opencosmo.units import UnitConvention
from opencosmo.units.handler import (
    make_unit_handler_from_hdf5,
    make_unit_handler_from_units,
)

if TYPE_CHECKING:
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
        raw_data_handler: DataHandler,
        cache: DataCache,
        derived_columns: dict[str, ConstructedColumn],
        unit_handler: UnitHandler,
        header: OpenCosmoHeader,
        columns: set[str],
        region: Region,
        sort_by: Optional[tuple[str, bool]],
    ):
        self.__raw_data_handler = raw_data_handler
        self.__cache = cache
        self.__derived_columns = derived_columns
        self.__unit_handler = unit_handler
        self.__header = header
        self.__columns = columns
        self.__region = region
        self.__sort_by = sort_by
        self.__cache.register_column_group(id(self), self.__columns)
        finalize(self, deregister_state, id(self), self.__cache)

    def __rebuild(self, **updates):
        new = {
            "raw_data_handler": self.__raw_data_handler,
            "cache": self.__cache,
            "derived_columns": self.__derived_columns,
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

        columns = set(handler.columns)
        cache = ColumnCache.empty()
        return DatasetState(
            handler,
            cache,
            {},
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
        descriptions: Optional[dict[str, str]] = {},
        index: Optional[DataIndex] = None,
    ):
        cache = ColumnCache.empty()
        cache.add_data(
            data_columns | metadata_columns, descriptions, set(metadata_columns.keys())
        )
        units: dict[str, u.Unit] = {}
        for name, column in data_columns.items():
            units[name] = None
            if isinstance(column, u.Quantity):
                units[name] = column.unit

        unit_handler = make_unit_handler_from_units(units, header, unit_convention)

        return DatasetState(
            EmptyHandler(),
            cache,
            {},
            unit_handler,
            header,
            set(data_columns.keys()),
            region,
            None,
        )

    def __len__(self):
        if isinstance(self.__raw_data_handler, EmptyHandler):
            return len(self.__cache)
        return len(self.__raw_data_handler)

    @property
    def descriptions(self):
        all_descriptions = (
            {name: col.description for name, col in self.__derived_columns.items()}
            | self.__raw_data_handler.descriptions
            | self.__cache.descriptions
        )
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
        return list(self.__columns)

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
        data = self.__build_derived_columns(unit_kwargs)
        cached_data = self.__cache.get_data(self.columns)
        converted_cached_data = self.__unit_handler.apply_unit_conversions(
            cached_data, unit_kwargs
        )

        data |= cached_data
        if converted_cached_data:
            self.__cache.add_data(converted_cached_data, {}, push_up=False)
            data |= converted_cached_data

        raw_columns = (
            set(self.columns)
            .intersection(self.__raw_data_handler.columns)
            .difference(data.keys())
        )
        if (
            self.__sort_by is not None
            and self.__sort_by[0] in self.__raw_data_handler.columns
        ):
            raw_columns.add(self.__sort_by[0])

        if raw_columns:
            raw_data = self.__raw_data_handler.get_data(raw_columns)
            raw_data = self.__unit_handler.apply_raw_units(raw_data, unit_kwargs)

            updated_data = self.__unit_handler.apply_unit_conversions(
                raw_data, unit_kwargs
            )
            new_data = raw_data | updated_data
            if new_data:
                self.__cache.add_data(new_data, {}, push_up=False)
            data |= new_data

        if missing := set(self.columns).difference(data.keys()):
            raise RuntimeError(
                f"Some columns are missing from the output! This is likely a bug. Please report it on GitHub. Missing: {missing}"
            )

        # keep ordering

        if metadata_columns:
            metadata = self.__cache.get_data(metadata_columns)
            additional_metadata_columns_to_fetch = set(metadata_columns).difference(
                metadata.keys()
            )
            metadata |= (
                self.__raw_data_handler.get_metadata(
                    additional_metadata_columns_to_fetch
                )
                or {}
            )

            data.update(metadata)

        if not ignore_sort and self.__sort_by is not None:
            sort_by = data[self.__sort_by[0]]
            order = np.argsort(sort_by)
            if self.__sort_by[1]:
                order = order[::-1]

            data = {key: value[order] for key, value in data.items()}

        new_order = [c for c in self.columns]
        if metadata_columns:
            new_order.extend(metadata_columns)

        return {name: data[name] for name in new_order}

    def rows(self, metadata_columns: list = [], unit_kwargs: dict = {}):
        derived_to_collect = (
            set(self.__derived_columns.keys())
            .intersection(self.columns)
            .difference(self.__cache.columns)
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
                self.__cache.add_data(data, {})
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
        index = np.where(mask)[0]
        new_raw_handler = self.__raw_data_handler.take(index)
        new_cache = self.__cache.take(index)
        return self.__rebuild(
            cache=new_cache,
            raw_data_handler=new_raw_handler,
        )

    def make_schema(self, name: Optional[str] = None):
        header = self.__header.with_region(self.__region)
        raw_columns = self.__columns.intersection(self.__raw_data_handler.columns)

        data_schema, metadata_schema = self.__raw_data_handler.make_schema(
            raw_columns, header
        )
        derived_names = set(self.__derived_columns.keys()).intersection(self.columns)
        derived_data = (
            self.select(derived_names)
            .with_units(self.__unit_handler.base_convention, {}, {}, None, None)
            .get_data(ignore_sort=True)
        )
        cached_data_schema, cached_metadata_schema = self.__cache.make_schema(
            self.columns + self.meta_columns
        )

        for colname in derived_names:
            if colname in cached_data_schema.columns:
                continue
            coldata = derived_data[colname]
            unit = ""
            if isinstance(coldata, u.Quantity):
                unit = str(coldata.unit)
                coldata = derived_data[colname].value

            attrs = {
                "unit": unit,
                "description": str(self.__derived_columns[colname].description),
            }
            source = NumpySource(coldata)
            writer = ColumnWriter([source], ColumnCombineStrategy.CONCAT, attrs=attrs)
            data_schema.columns[colname] = writer

        attributes = {}
        if (load_conditions := self.__raw_data_handler.load_conditions) is not None:
            attributes["load/if"] = load_conditions

        data_schema = combine_with_cached_schema(data_schema, cached_data_schema)

        metadata_schema = combine_with_cached_schema(
            metadata_schema, cached_metadata_schema
        )
        children = {"data": data_schema}

        if metadata_schema.type != FileEntry.EMPTY:
            children[metadata_schema.name] = metadata_schema
        if name is None:
            name = ""

        return make_schema(
            name, FileEntry.DATASET, children=children, attributes=attributes
        )

    def with_new_columns(
        self,
        descriptions: dict[str, str] = {},
        **new_columns: ConstructedColumn | np.ndarray | u.Quantity,
    ):
        """
        Add a set of derived columns to the dataset. A derived column is a column that
        has been created based on the values in another column.
        """

        existing_columns = set(self.columns)

        if inter := existing_columns.intersection(new_columns.keys()):
            raise ValueError(f"Some columns are already in the dataset: {inter}")

        new_derived_columns = {}
        new_in_memory_columns = {}
        new_in_memory_descriptions = {}

        for colname, column in new_columns.items():
            match column:
                case DerivedColumn() | EvaluatedColumn() | Column():
                    column.description = descriptions.get(colname, "None")
                    new_derived_columns[colname] = column
                case np.ndarray():
                    if len(column) != len(self):
                        raise ValueError(
                            f"New column {colname} does not have the same length as this dataset!"
                        )
                    new_in_memory_descriptions[colname] = descriptions.get(
                        colname, "None"
                    )
                    new_in_memory_columns[colname] = column
                case _:
                    raise ValueError(
                        f"Got an invalid new column of type {type(column)}"
                    )

        new_unit_handler = self.__unit_handler
        new_derived = copy(self.__derived_columns)
        new_column_names: set[str] = set(self.columns)
        if new_in_memory_columns:
            new_unit_handler = validate_in_memory_columns(
                new_in_memory_columns, self.__unit_handler, len(self)
            )
            new_in_memory_columns = resort(
                new_in_memory_columns, self.get_sorted_index()
            )
            self.__cache.add_data(
                new_in_memory_columns, descriptions=new_in_memory_descriptions
            )
            new_column_names |= set(new_in_memory_columns.keys())

        if new_derived_columns:
            new_units = validate_derived_columns(
                self.__derived_columns | new_derived_columns,
                existing_columns.union(new_in_memory_columns.keys()).difference(
                    self.__derived_columns.keys()
                ),
                new_unit_handler.base_units,
            )
            new_derived |= new_derived_columns
            for colname, derived in new_derived.items():
                if (prod := derived.produces) is not None:
                    new_column_names |= prod
                else:
                    new_column_names.add(colname)

            new_unit_handler = new_unit_handler.with_new_columns(**new_units)

        return self.__rebuild(
            cache=self.__cache,
            derived_columns=new_derived,
            columns=new_column_names,
            unit_handler=new_unit_handler,
        )

    def __build_derived_columns(self, unit_kwargs: dict) -> table.Table:
        """
        Build any derived columns that are present in this dataset
        """
        if not self.__derived_columns:
            return {}

        all_derived_columns: set[str] = reduce(
            lambda acc, dc: acc.union(
                dc[1].produces if dc[1].produces is not None else {dc[0]}
            ),
            self.__derived_columns.items(),
            set(),
        )
        derived_names = all_derived_columns.intersection(self.columns)
        if self.__sort_by is not None and self.__sort_by[0] in all_derived_columns:
            derived_names.add(self.__sort_by[0])

        dc = build_derived_columns(
            self.__derived_columns,
            derived_names,
            self.__cache,
            self.__raw_data_handler,
            self.__unit_handler,
            unit_kwargs,
            self.__raw_data_handler.index,
        )
        return dc

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

        return self.__rebuild(columns=selections)

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

        take_index: DataIndex

        if at == "start":
            return self.take_range(0, n)
        elif at == "end":
            return self.take_range(len(self) - n, len(self))
        elif at == "random":
            row_indices = np.random.choice(len(self), n, replace=False)
            row_indices.sort()

        sorted = self.get_sorted_index()
        if sorted is None:
            take_index = row_indices
        else:
            take_index = np.sort(sorted[row_indices])

        new_handler = self.__raw_data_handler.take(take_index)
        new_cache = self.__cache.take(take_index)

        return self.__rebuild(
            raw_data_handler=new_handler,
            cache=new_cache,
        )

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

        sorted = self.get_sorted_index()
        take_index: DataIndex
        if sorted is None:
            take_index = single_chunk(start, end - start)
        else:
            take_index = np.sort(sorted[start:end])

        new_raw_handler = self.__raw_data_handler.take(take_index)
        new_im = self.__cache.take(take_index)
        return self.__rebuild(
            raw_data_handler=new_raw_handler,
            cache=new_im,
        )

    def take_rows(self, rows: DataIndex):
        if len(self) == 0:
            return self
        row_range = get_range(rows)

        if row_range[1] > len(self) or row_range[0] < 0:
            raise ValueError(
                "Row indices must be between 0 and the length of this dataset!"
            )
        sorted = self.get_sorted_index()
        new_handler = self.__raw_data_handler.take(rows, sorted)
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

        original_units = self.__unit_handler.current_units
        new_units = new_handler.current_units
        cached_columns_to_drop: Iterable[str]

        if convention_ == self.__unit_handler.current_convention:
            cached_columns_to_drop = [
                name for name in new_units if original_units[name] != new_units[name]
            ]
        else:
            cached_columns_to_drop = new_units.keys()
        all_derived_names: set[str] = reduce(
            lambda acc, next: acc.union(next.produces or set()),
            self.__derived_columns.values(),
            set(),
        )
        cached_columns_to_drop = list(
            filter(
                lambda name: name in self.__raw_data_handler.columns
                or name in all_derived_names,
                cached_columns_to_drop,
            )
        )

        return self.__rebuild(
            unit_handler=new_handler, cache=self.__cache.drop(cached_columns_to_drop)
        )
