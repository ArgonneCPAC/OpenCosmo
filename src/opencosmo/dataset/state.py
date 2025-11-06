from __future__ import annotations

from functools import reduce
from itertools import cycle
from typing import TYPE_CHECKING, Iterable, Optional
from weakref import finalize

import astropy.units as u
import numpy as np
from astropy.table import QTable

from opencosmo.column.cache import ColumnCache
from opencosmo.column.column import DerivedColumn
from opencosmo.dataset.handler import Hdf5Handler
from opencosmo.index import ChunkedIndex, SimpleIndex
from opencosmo.io import schemas as ios
from opencosmo.units import UnitConvention
from opencosmo.units.handler import make_unit_handler

if TYPE_CHECKING:
    import h5py
    from astropy import table, units
    from astropy.cosmology import Cosmology
    from numpy.typing import NDArray

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.spatial.protocols import Region
    from opencosmo.units.handler import UnitHandler


def deregister_state(id: int, cache: ColumnCache):
    cache.deregister_column_group(id)


class DatasetState:
    """
    Holds mutable state required by the dataset. Cleans up the dataset to mostly focus
    on very high-level operations. Not a user facing class.
    """

    def __init__(
        self,
        raw_data_handler: Hdf5Handler,
        cache: ColumnCache,
        derived_columns: dict[str, DerivedColumn],
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
    def from_group(
        cls,
        group: h5py.Group,
        header: OpenCosmoHeader,
        unit_convention: UnitConvention,
        region: Region,
        index: Optional[DataIndex] = None,
        metadata_group: Optional[h5py.Group] = None,
    ):
        handler = Hdf5Handler.from_group(group, index, metadata_group)
        unit_handler = make_unit_handler(handler.data, header, unit_convention)

        columns = set(handler.columns)
        cache = ColumnCache.empty()
        return DatasetState(
            handler,
            cache,
            {},
            unit_handler,
            header,
            columns,
            region,
            None,
        )

    def __len__(self):
        return len(self.__raw_data_handler.index)

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
        return self.__raw_data_handler.index

    @property
    def unit_handler(self):
        return self.__unit_handler

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
        return self.__raw_data_handler.metadata_columns

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
        cached_data = self.__cache.get_columns(self.columns)
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

            self.__cache.add_data(raw_data)
            updated_data = self.__unit_handler.apply_unit_conversions(
                raw_data, unit_kwargs
            )
            if updated_data:
                self.__cache.add_data(updated_data, push_up=False)
            data |= raw_data | updated_data

        if not set(data.keys()).issuperset(self.columns):
            raise RuntimeError(
                "Some columns are missing from the output! This is likely a bug. Please report it on GitHub"
            )

        # keep ordering
        output = QTable(data, copy=False)

        data_columns = set(output.columns)

        if metadata_columns:
            output.update(self.__raw_data_handler.get_metadata(metadata_columns))

        if not ignore_sort and self.__sort_by is not None:
            order = output.argsort(self.__sort_by[0], reverse=self.__sort_by[1])
            output = output[order]

        new_order = [c for c in self.columns]
        if metadata_columns:
            new_order.extend(metadata_columns)

        return output[new_order]

    def get_metadata(self, columns=[]):
        return self.__raw_data_handler.get_metadata(columns)

    def with_mask(self, mask: NDArray[np.bool_]):
        index = SimpleIndex(np.where(mask)[0])
        new_raw_handler = self.__raw_data_handler.mask(mask)
        new_cache = self.__cache.take(index)
        return self.__rebuild(
            cache=new_cache,
            raw_data_handler=new_raw_handler,
        )

    def make_schema(self):
        header = self.__header.with_region(self.__region)
        raw_columns = self.__columns.intersection(self.__raw_data_handler.columns)

        schema = self.__raw_data_handler.make_schema(raw_columns, header)
        derived_names = set(self.__derived_columns.keys()).intersection(self.columns)
        derived_data = (
            self.select(derived_names)
            .with_units("unitless", {}, {}, None, None)
            .get_data(ignore_sort=True)
        )
        column_units = {
            name: self.__unit_handler.base_units[name] for name in raw_columns
        }

        for colname in derived_names:
            unit = self.__derived_columns[colname].get_units(column_units)
            attrs = {
                "unit": str(unit),
                "description": self.__derived_columns[colname].description,
            }
            coldata = derived_data[colname].value
            colschema = ios.ColumnSchema(
                colname, ChunkedIndex.from_size(len(coldata)), coldata, attrs
            )
            schema.add_child(colschema, f"data/{colname}")

        cached_data = self.__cache.get_columns(self.columns)

        for colname, coldata in cached_data.items():
            try:
                data = coldata.value
                unit_str = str(coldata.unit)
            except AttributeError:
                data = coldata
                unit_str = ""
            if colname in schema.columns["data"]:
                continue

            attrs = {}
            attrs["unit"] = unit_str
            attrs["description"] = self.descriptions.get(colname, "None")

            colschema = ios.ColumnSchema(
                colname, ChunkedIndex.from_size(len(coldata)), data, attrs
            )
            schema.add_child(colschema, f"data/{colname}")

        return schema

    def with_new_columns(
        self,
        descriptions: dict[str, str] = {},
        **new_columns: DerivedColumn | np.ndarray | units.Quantity,
    ):
        """
        Add a set of derived columns to the dataset. A derived column is a column that
        has been created based on the values in another column.
        """
        derived_update: dict[str, DerivedColumn] = {}
        new_unit_handler = self.__unit_handler

        if inter := set(self.columns).intersection(new_columns.keys()):
            raise ValueError(f"Some columns are already in the dataset: {inter}")

        new_column_names = self.__columns.copy()
        new_derived = {}
        new_in_memory = {}
        new_in_memory_descriptions = {}
        new_units = {}
        for colname, column in new_columns.items():
            description = descriptions.get(colname, "None")
            match column:
                case DerivedColumn():
                    ancestor_columns = column.requires()
                    missing = ancestor_columns.difference(ancestor_columns)
                    if missing:
                        raise ValueError(
                            f"Missing columns {missing} required for derived column {colname}"
                        )

                    derived_column_unit = column.get_units(
                        self.__unit_handler.base_units
                    )
                    new_units[colname] = derived_column_unit
                    column.description = description
                    new_derived[colname] = column
                    new_column_names.add(colname)
                case np.ndarray():
                    if len(column) != len(self):
                        raise ValueError(
                            f"In-memory columns must have the same length as the dataset!"
                        )
                    new_in_memory[colname] = column
                    new_column_names.add(colname)
                    new_in_memory_descriptions[colname] = description
                    new_units[colname] = None
                    if isinstance(column, u.Quantity):
                        new_units[colname] = column.unit
                case _:
                    raise ValueError(f"Unexpected new column type: {type(column)}")
        if new_units:
            new_unit_handler = new_unit_handler.with_new_columns(**new_units)

        new_derived = self.__derived_columns | new_derived
        new_cache = self.__cache
        if new_in_memory:
            new_cache = self.__cache.with_data(
                new_in_memory, descriptions=new_in_memory_descriptions
            )
        return self.__rebuild(
            cache=new_cache,
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

        derived_names = set(self.__derived_columns.keys()).intersection(self.columns)
        if (
            self.__sort_by is not None
            and self.__sort_by[0] in self.__derived_columns.keys()
        ):
            derived_names.add(self.__sort_by[0])

        ancestors: set[str] = reduce(
            lambda acc, der: acc.union(der.requires()),
            self.__derived_columns.values(),
            set(),
        )
        ad = ancestors.intersection(self.__derived_columns.keys())
        while ad:
            derived_names = derived_names.union(ad)
            for col in ad:
                ancestors.remove(col)
                ancestors = ancestors.union(self.__derived_columns[col].requires())
            ad = ancestors.intersection(derived_names)

        cached_data = self.__cache.get_columns(ancestors)
        remaining_ancestors = ancestors.difference(cached_data.keys())
        raw_ancestors = ancestors.intersection(remaining_ancestors)

        raw_data = self.__raw_data_handler.get_data(raw_ancestors)
        data = cached_data | self.__unit_handler.apply_units(raw_data, unit_kwargs)
        seen: set[str] = set()

        for name in cycle(derived_names):
            if derived_names.issubset(data.keys()):
                break
            elif name in seen:
                # We're stuck in a loop
                raise ValueError(
                    "Something went wrong when trying to instatiate derived columns!"
                )
            elif name in data:
                continue
            elif set(data.keys()).issuperset(self.__derived_columns[name].requires()):
                data[name] = self.__derived_columns[name].evaluate(data)
                seen = set()
            else:
                seen.add(name)

        return data

    def __get_im_columns(self, data: dict, unit_kwargs) -> table.Table:
        im_data = {}
        for colname, column in self.__cache.columns():
            im_data[colname] = column

        return self.__unit_handler.apply_units(im_data, unit_kwargs)

    def with_region(self, region: Region):
        """
        Return the same dataset but with a different region
        """
        return self.__rebuild(region=region)

    def select(self, columns: str | Iterable[str]):
        """
        Select a subset of columns from the dataset. It is possible for a user to select
        a derived column in the dataset, but not the columns it is derived from.
        This class tracks any columns which are required to materialize the dataset but
        are not in the final selection in self.__hidden. When the dataset is
        materialized, the columns in self.__hidden are removed before the data is
        returned to the user.

        """
        if isinstance(columns, str):
            columns = [columns]

        columns = set(columns)
        missing = columns - self.__columns
        if missing:
            raise ValueError(
                f"Tried to select columns that are not in this dataset: {missing}"
            )

        return self.__rebuild(columns=columns)

    def sort_by(self, column_name: str, invert: bool):
        if column_name not in self.columns:
            raise ValueError(f"This dataset has no column {column_name}")

        return self.__rebuild(sort_by=(column_name, invert))

    def get_sorted_index(self):
        if self.__sort_by is not None:
            column = self.select(self.__sort_by[0]).get_data(ignore_sort=True)[
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

        sorted = self.get_sorted_index()
        if sorted is None:
            take_index = SimpleIndex(row_indices)
        else:
            take_index = SimpleIndex(np.sort(sorted[row_indices]))

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
        if end > len(self.__raw_data_handler.index):
            raise ValueError("end must be less than the length of the dataset.")

        if start < 0 or end > len(self.__raw_data_handler.index):
            raise ValueError("start and end must be within the bounds of the dataset.")

        sorted = self.get_sorted_index()
        take_index: DataIndex
        if sorted is None:
            take_index = ChunkedIndex.single_chunk(start, end - start)
        else:
            take_index = SimpleIndex(np.sort(sorted[start:end]))

        new_raw_handler = self.__raw_data_handler.take(take_index)
        new_im = self.__cache.take(take_index)
        return self.__rebuild(
            raw_data_handler=new_raw_handler,
            cache=new_im,
        )

    def take_rows(self, rows: DataIndex):
        if len(self) == 0:
            return self
        if rows.range()[1] > len(self) or rows.range()[0] < 0:
            raise ValueError(
                f"Row indices must be between 0 and the length of this dataset!"
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
            cache = self.__cache.duplicate()
        else:
            convention_ = UnitConvention(convention)
            cache = self.__cache.without_columns(self.__raw_data_handler.columns)
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
        return self.__rebuild(unit_handler=new_handler, cache=cache)
