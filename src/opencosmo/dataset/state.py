from functools import reduce
from typing import TYPE_CHECKING, Iterable, Optional

import astropy.units as u
import numpy as np
from astropy import table, units  # type: ignore
from astropy.cosmology import Cosmology  # type: ignore
from astropy.table import QTable
from numpy.typing import NDArray

from opencosmo.dataset.column import DerivedColumn
from opencosmo.dataset.im import InMemoryColumnHandler
from opencosmo.header import OpenCosmoHeader
from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex
from opencosmo.io import schemas as ios
from opencosmo.spatial.protocols import Region
from opencosmo.units import UnitConvention
from opencosmo.units.handler import UnitHandler

if TYPE_CHECKING:
    from opencosmo.dataset.handler import DatasetHandler


class DatasetState:
    """
    Holds mutable state required by the dataset. Cleans up the dataset to mostly focus
    on very high-level operations. Not a user facing class.
    """

    def __init__(
        self,
        unit_handler: UnitHandler,
        index: DataIndex,
        columns: set[str],
        region: Region,
        header: OpenCosmoHeader,
        im_handler: InMemoryColumnHandler,
        sort_by: Optional[tuple[str, bool]] = None,
        hidden: set[str] = set(),
        derived: dict[str, DerivedColumn] = {},
    ):
        self.__unit_handler = unit_handler
        self.__im_handler = im_handler
        self.__columns = columns
        self.__derived: dict[str, DerivedColumn] = derived
        self.__header = header
        self.__hidden = hidden
        self.__index = index
        self.__sort_by = sort_by
        self.__region = region

    @property
    def index(self):
        return self.__index

    @property
    def descriptions(self):
        return self.__im_handler.descriptions | {
            name: col.description for name, col in self.__derived.items()
        }

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
        columns = (
            set(self.__columns)
            | set(self.__derived.keys())
            | set(self.__im_handler.keys())
        )
        return list(columns - self.__hidden)

    def get_data(
        self,
        handler: "DatasetHandler",
        ignore_sort: bool = False,
        attach_index: bool = False,
        unit_kwargs: dict = {},
    ) -> table.QTable:
        """
        Get the data for a given handler.
        """
        data = handler.get_data(self.__columns, index=self.__index)
        data = self.__get_im_columns(data)
        data = self.__build_derived_columns(data)
        data = self.__unit_handler.apply_units(data, unit_kwargs)
        output = QTable(data)

        data_columns = set(output.columns)
        index_array = self.__index.into_array()

        if not ignore_sort and self.__sort_by is not None:
            order = output.argsort(self.__sort_by[0], reverse=self.__sort_by[1])
            output = output[order]
            index_array = index_array[order]
        if (
            self.__hidden
            and not self.__hidden.intersection(data_columns) == data_columns
        ):
            output.remove_columns(self.__hidden)
        if attach_index:
            output["raw_index"] = index_array
        return output

    def with_index(self, index: DataIndex):
        """
        Return the same dataset state with a new index
        """

        new_cache = self.__im_handler.project(index)

        return DatasetState(
            self.__unit_handler,
            index,
            self.__columns,
            self.__region,
            self.__header,
            new_cache,
            self.__sort_by,
            self.__hidden,
            self.__derived,
        )

    def with_mask(self, mask: NDArray[np.bool_]):
        new_index = self.__index.mask(mask)
        return self.with_index(new_index)

    def make_schema(self, handler: "DatasetHandler"):
        header = self.__header.with_region(self.__region)
        schema = handler.prep_write(
            self.__index, self.__columns - self.__hidden, header
        )
        derived_names = set(self.__derived.keys()) - self.__hidden
        derived_data = (
            self.select(derived_names)
            .with_units("unitless", {}, {}, None, None)
            .get_data(handler)
        )
        column_units = {
            name: self.__unit_handler.base_units[name] for name in self.columns
        }

        for colname in derived_names:
            attrs = {"unit": str(column_units[colname])}
            coldata = derived_data[colname].value
            colschema = ios.ColumnSchema(
                colname, ChunkedIndex.from_size(len(coldata)), coldata, attrs
            )
            schema.add_child(colschema, colname)

        for colname, coldata in self.__im_handler.columns():
            if colname in self.__hidden:
                continue
            attrs = {}
            attrs["unit"] = str(column_units.get(colname))

            colschema = ios.ColumnSchema(
                colname, ChunkedIndex.from_size(len(coldata)), coldata, attrs
            )
            schema.add_child(colschema, colname)

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
        column_names = (
            set(self.__columns)
            | set(self.__derived.keys())
            | set(self.__im_handler.keys())
        )
        new_im_handler = self.__im_handler
        derived_update = {}
        new_unit_handler = self.__unit_handler
        for name, new_col in new_columns.items():
            if name in column_names:
                raise ValueError(f"Dataset already has column named {name}")

            if isinstance(new_col, np.ndarray):
                if len(new_col) != len(self.__index):
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

                new_im_handler = new_im_handler.with_new_column(name, new_col)

            elif not new_col.check_parent_existance(column_names):
                raise ValueError(
                    f"Column {name} is derived from columns "
                    "that are not in the dataset!"
                )
            else:
                unit = new_col.get_units(self.__unit_handler.base_units)
                new_unit_handler = new_unit_handler.with_new_columns(**{name: unit})
                derived_update[name] = new_col
            column_names.add(name)

        new_derived = self.__derived | derived_update
        return DatasetState(
            new_unit_handler,
            self.__index,
            self.__columns,
            self.__region,
            self.__header,
            new_im_handler,
            self.__sort_by,
            self.__hidden,
            new_derived,
        )

    def __build_derived_columns(self, data: table.Table) -> table.Table:
        """
        Build any derived columns that are present in this dataset
        """
        for colname, column in self.__derived.items():
            new_column = column.evaluate(data)
            data[colname] = new_column
        return data

    def __get_im_columns(self, data: dict) -> table.Table:
        for colname, column in self.__im_handler.columns():
            data[colname] = column
        return data

    def with_region(self, region: Region):
        """
        Return the same dataset but with a different region
        """
        return DatasetState(
            self.__unit_handler,
            self.__index,
            self.__columns,
            region,
            self.__header,
            self.__im_handler,
            self.__sort_by,
            self.__hidden,
            self.__derived,
        )

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
        new_hidden = set()
        if self.__sort_by is not None:
            if self.__sort_by[0] not in columns:
                new_hidden.add(self.__sort_by[0])
            columns.add(self.__sort_by[0])

        known_raw = self.__columns
        known_derived = set(self.__derived.keys())
        known_im = set(self.__im_handler.keys())
        unknown_columns = columns - known_raw - known_derived - known_im
        if unknown_columns:
            raise ValueError(
                "Tried to select columns that aren't in this dataset! Missing columns "
                + ", ".join(unknown_columns)
            )

        required_derived = known_derived.intersection(columns)
        required_raw = known_raw.intersection(columns)
        required_im = known_im.intersection(columns)

        additional_derived = required_derived

        while additional_derived:
            # Follow any chains of derived columns until we reach columns that are
            # actually in the raw data.
            required_derived |= additional_derived
            additional_columns: set[str] = reduce(
                lambda s, derived: s.union(self.__derived[derived].requires()),
                additional_derived,
                set(),
            )
            required_raw |= additional_columns.intersection(known_raw)
            required_im |= additional_columns.intersection(known_im)
            additional_derived = additional_columns.intersection(known_derived)

        all_required = required_derived | required_raw | required_im

        # Derived columns have to be instantiated in the order they are created in order
        # to ensure chains of derived columns work correctly
        new_derived = {k: v for k, v in self.__derived.items() if k in required_derived}
        # Builders can be performed in any order
        new_im_handler = self.__im_handler.with_columns(required_im)

        new_hidden.update(all_required - columns)
        if self.__sort_by is not None and self.__sort_by[0] not in columns:
            new_hidden.add(self.__sort_by[0])

        return DatasetState(
            self.__unit_handler,
            self.__index,
            required_raw,
            self.__region,
            self.__header,
            new_im_handler,
            self.__sort_by,
            new_hidden,
            new_derived,
        )

    def sort_by(self, column_name: str, handler: "DatasetHandler", invert: bool):
        return DatasetState(
            self.__unit_handler,
            self.__index,
            self.__columns,
            self.__region,
            self.__header,
            self.__im_handler,
            (column_name, invert),
            self.__hidden,
            self.__derived,
        )

    def take(self, n: int, at: str, handler):
        """
        Take rows from the dataset.
        """

        if self.__sort_by is not None:
            column = self.select(self.__sort_by[0]).get_data(handler, ignore_sort=True)[
                self.__sort_by[0]
            ]
            sorted = np.argsort(column)
            if self.__sort_by[1]:
                sorted = sorted[::-1]

            index: DataIndex = SimpleIndex(sorted)
        else:
            index = self.__index

        new_index = index.take(n, at)
        if self.__sort_by is not None:
            new_idxs = self.__index.into_array()[new_index.into_array()]
            new_index = SimpleIndex(np.sort(new_idxs))

        return self.with_index(new_index)

    def take_range(self, start: int, end: int):
        """
        Take a range of rows form the dataset.
        """
        if start < 0 or end < 0:
            raise ValueError("start and end must be positive.")
        if end < start:
            raise ValueError("end must be greater than start.")
        if end > len(self.__index):
            raise ValueError("end must be less than the length of the dataset.")

        if start < 0 or end > len(self.__index):
            raise ValueError("start and end must be within the bounds of the dataset.")

        new_index = self.__index.take_range(start, end)
        return self.with_index(new_index)

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

        return DatasetState(
            self.__unit_handler.with_convention(convention_).with_conversions(
                conversions, columns
            ),
            self.__index,
            self.__columns,
            self.__region,
            self.__header,
            self.__im_handler,
            self.__sort_by,
            self.__hidden,
            self.__derived,
        )
