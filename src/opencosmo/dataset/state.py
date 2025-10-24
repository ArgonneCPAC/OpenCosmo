from __future__ import annotations

from functools import reduce
from itertools import cycle
from typing import TYPE_CHECKING, Iterable, Optional
from weakref import finalize

import astropy.units as u
import numpy as np
from astropy.table import QTable

from opencosmo.dataset.handler import Hdf5Handler
from opencosmo.dataset.im import InMemoryColumnHandler
from opencosmo.index import ChunkedIndex, SimpleIndex
from opencosmo.io import schemas as ios
from opencosmo.units import UnitConvention
from opencosmo.units.handler import make_unit_handler

if TYPE_CHECKING:
    import h5py
    from astropy import table, units
    from astropy.cosmology import Cosmology
    from numpy.typing import NDArray

    from opencosmo.dataset.column import DerivedColumn
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.spatial.protocols import Region
    from opencosmo.units.handler import UnitHandler


def deregister_state(id: int, handler: Hdf5Handler):
    handler.deregister(id)


class DatasetState:
    """
    Holds mutable state required by the dataset. Cleans up the dataset to mostly focus
    on very high-level operations. Not a user facing class.
    """

    def __init__(
        self,
        raw_data_handler: Hdf5Handler,
        im_handler: InMemoryColumnHandler,
        derived_columns: dict[str, DerivedColumn],
        unit_handler: UnitHandler,
        header: OpenCosmoHeader,
        columns: set[str],
        region: Region,
        sort_by: Optional[tuple[str, bool]],
    ):
        self.__raw_data_handler = raw_data_handler
        self.__im_handler = im_handler
        self.__derived_columns = derived_columns
        self.__unit_handler = unit_handler
        self.__header = header
        self.__columns = columns
        self.__region = region
        self.__sort_by = sort_by
        self.__raw_data_handler.register(self)
        finalize(self, deregister_state, id(self), self.__raw_data_handler)

    def __rebuild(self, **updates):
        new = {
            "raw_data_handler": self.__raw_data_handler,
            "im_handler": self.__im_handler,
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
        im_handler = InMemoryColumnHandler.empty()
        return DatasetState(
            handler,
            im_handler,
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
        return (
            self.__im_handler.descriptions
            | {name: col.description for name, col in self.__derived_columns.items()}
            | self.__raw_data_handler.descriptions
        )

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
        data = data | self.__get_im_columns(data, unit_kwargs)

        raw_columns = set(self.columns).difference(data.keys())
        if (
            self.__sort_by is not None
            and self.__sort_by[0] in self.__raw_data_handler.columns
        ):
            raw_columns.add(self.__sort_by[0])

        raw_data = self.__raw_data_handler.get_data(raw_columns)
        data = data | self.__unit_handler.apply_units(raw_data, unit_kwargs)

        output = QTable(data, copy=False)

        data_columns = set(output.columns)
        raw_index_array = self.__raw_data_handler.index.into_array()

        if metadata_columns:
            output.update(self.__raw_data_handler.get_metadata(metadata_columns))

        if not ignore_sort and self.__sort_by is not None:
            order = output.argsort(self.__sort_by[0], reverse=self.__sort_by[1])
            output = output[order]
            raw_index_array = raw_index_array[order]

        extras = set(data.keys()).difference(self.columns)
        if extras:
            output.remove_columns(extras)

        return output

    def get_metadata(self, columns=[]):
        return self.__raw_data_handler.get_metadata(columns)

    def with_mask(self, mask: NDArray[np.bool_]):
        new_raw_handler = self.__raw_data_handler.mask(mask)
        new_im_handler = self.__im_handler.get_rows(mask)
        return self.__rebuild(
            im_handler=new_im_handler,
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
            .get_data()
        )
        column_units = {
            name: self.__unit_handler.base_units[name] for name in self.columns
        }

        for colname in derived_names:
            attrs = {
                "unit": str(column_units[colname]),
                "description": self.__derived_columns[colname].description,
            }
            coldata = derived_data[colname].value
            colschema = ios.ColumnSchema(
                colname, ChunkedIndex.from_size(len(coldata)), coldata, attrs
            )
            schema.add_child(colschema, f"data/{colname}")

        im_names = set(self.__im_handler.keys()).intersection(self.columns)
        im_descriptions = self.__im_handler.descriptions
        for colname, coldata in self.__im_handler.columns():
            if colname not in im_names:
                continue
            attrs = {}
            attrs["unit"] = str(column_units.get(colname))
            attrs["description"] = im_descriptions.get(colname)

            colschema = ios.ColumnSchema(
                colname, ChunkedIndex.from_size(len(coldata)), coldata, attrs
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
        derived_update = {}
        new_unit_handler = self.__unit_handler
        new_im_handler = self.__im_handler
        for name, new_col in new_columns.items():
            if name in self.__columns:
                raise ValueError(f"Dataset already has column named {name}")

            if isinstance(new_col, np.ndarray):
                if len(new_col) != len(self):
                    raise ValueError(
                        f"New column {name} has length {len(new_col)} but this dataset "
                        "has length {len(self.__index)}"
                    )
                if isinstance(new_col, u.Quantity):
                    new_unit_handler = new_unit_handler.with_static_columns(
                        **{name: new_col.unit}
                    )
                    new_col = new_col.value
                else:
                    new_unit_handler = new_unit_handler.with_new_columns(**{name: None})

                new_im_handler = new_im_handler.with_new_column(
                    name, new_col, descriptions.get(name, "None")
                )

            elif not new_col.check_parent_existance(self.__columns):
                raise ValueError(
                    f"Column {name} is derived from columns "
                    "that are not in the dataset!"
                )
            else:
                unit = new_col.get_units(self.__unit_handler.base_units)
                new_col.description = descriptions.get(name, "None")
                new_unit_handler = new_unit_handler.with_new_columns(**{name: unit})
                derived_update[name] = new_col

        new_derived = self.__derived_columns | derived_update
        new_columns_ = self.__columns.union(new_columns.keys())
        return self.__rebuild(
            unit_handler=new_unit_handler,
            im_handler=new_im_handler,
            derived_columns=new_derived,
            columns=new_columns_,
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
        raw_ancestors = ancestors.intersection(self.__raw_data_handler.columns)
        im_ancestors = ancestors.intersection(self.__im_handler.keys())
        additional_derived = ancestors.intersection(self.__derived_columns.keys())
        derived_names = derived_names.union(additional_derived)

        data = self.__raw_data_handler.get_data(raw_ancestors)
        data = data | self.__im_handler.get_data(im_ancestors)
        data = self.__unit_handler.apply_units(data, unit_kwargs)
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
        for colname, column in self.__im_handler.columns():
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

        sorted = self.get_sorted_index()
        take_index: DataIndex

        if at == "start":
            return self.take_range(0, n)
        elif at == "end":
            return self.take_range(len(self) - n, len(self))
        elif at == "random":
            row_indices = np.random.choice(len(self), n, replace=False)
            take_index = SimpleIndex(row_indices)

        new_handler = self.__raw_data_handler.take(take_index, sorted)
        new_im_handler = self.__im_handler.take(take_index, sorted)

        return self.__rebuild(
            raw_data_handler=new_handler,
            im_handler=new_im_handler,
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

        take_index = ChunkedIndex.single_chunk(start, end - start)

        new_raw_handler = self.__raw_data_handler.take(take_index, sorted)
        new_im = self.__im_handler.take(take_index, sorted)
        return self.__rebuild(
            raw_data_handler=new_raw_handler,
            im_handler=new_im,
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
        new_im_handler = self.__im_handler.take(rows, sorted)

        return self.__rebuild(
            raw_data_handler=new_handler,
            im_handler=new_im_handler,
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
        return self.__rebuild(unit_handler=new_handler)
