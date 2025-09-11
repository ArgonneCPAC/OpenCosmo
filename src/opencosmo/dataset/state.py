from functools import reduce
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from astropy import table, units  # type: ignore
from astropy.cosmology import Cosmology  # type: ignore
from numpy.typing import NDArray

import opencosmo.transformations.units as u
from opencosmo.dataset.builders import TableBuilder, get_table_builder
from opencosmo.dataset.column import DerivedColumn
from opencosmo.dataset.im import InMemoryColumnHandler
from opencosmo.header import OpenCosmoHeader
from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex
from opencosmo.io import schemas as ios
from opencosmo.spatial.protocols import Region

if TYPE_CHECKING:
    from opencosmo.dataset.handler import DatasetHandler


class DatasetState:
    """
    Holds mutable state required by the dataset. Cleans up the dataset to mostly focus
    on very high-level operations. Not a user facing class.
    """

    def __init__(
        self,
        base_unit_transformations: dict,
        builder: TableBuilder,
        index: DataIndex,
        convention: u.UnitConvention,
        region: Region,
        header: OpenCosmoHeader,
        im_handler: InMemoryColumnHandler,
        sort_by: Optional[tuple[str, bool]] = None,
        hidden: set[str] = set(),
        derived: dict[str, DerivedColumn] = {},
    ):
        self.__base_unit_transformations = base_unit_transformations
        self.__builder = builder
        self.__im_handler = im_handler
        self.__convention = convention
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
    def builder(self):
        return self.__builder

    @property
    def convention(self):
        return self.__convention

    @property
    def region(self):
        return self.__region

    @property
    def header(self):
        return self.__header

    @property
    def columns(self) -> list[str]:
        columns = (
            set(self.__builder.columns)
            | set(self.__derived.keys())
            | set(self.__im_handler.keys())
        )
        return list(columns - self.__hidden)

    def get_data(
        self,
        handler: "DatasetHandler",
        ignore_sort: bool = False,
        attach_index: bool = False,
    ) -> table.QTable:
        """
        Get the data for a given handler.
        """
        data = handler.get_data(builder=self.__builder, index=self.__index)
        data = self.__get_im_columns(data)
        data = self.__build_derived_columns(data)
        data_columns = set(data.columns)
        index_array = self.__index.into_array()

        if not ignore_sort and self.__sort_by is not None:
            order = data.argsort(self.__sort_by[0], reverse=self.__sort_by[1])
            data = data[order]
            index_array = index_array[order]
        if (
            self.__hidden
            and not self.__hidden.intersection(data_columns) == data_columns
        ):
            data.remove_columns(self.__hidden)
        if attach_index:
            data["raw_index"] = index_array
        return data

    def with_index(self, index: DataIndex):
        """
        Return the same dataset state with a new index
        """

        new_cache = self.__im_handler.project(index)

        return DatasetState(
            self.__base_unit_transformations,
            self.__builder,
            index,
            self.__convention,
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
        builder_names = set(self.__builder.columns)
        header = self.__header.with_region(self.__region)
        schema = handler.prep_write(self.__index, builder_names - self.__hidden, header)
        derived_names = set(self.__derived.keys()) - self.__hidden
        derived_data = (
            self.select(derived_names)
            .with_units("unitless", None, None)
            .get_data(handler)
        )

        column_units = handler.get_raw_units(builder_names)
        for dn, derived in self.__derived.items():
            column_units[dn] = derived.get_units(column_units)

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
            if isinstance(coldata, units.Quantity):
                attrs["unit"] = str(coldata.unit)
                coldata = coldata.value

            colschema = ios.ColumnSchema(
                colname, ChunkedIndex.from_size(len(coldata)), coldata, attrs
            )
            schema.add_child(colschema, colname)

        return schema

    def with_new_columns(
        self, **new_columns: DerivedColumn | np.ndarray | units.Quantity
    ):
        """
        Add a set of derived columns to the dataset. A derived column is a column that
        has been created based on the values in another column.
        """
        column_names = (
            set(self.builder.columns)
            | set(self.__derived.keys())
            | set(self.__im_handler.keys())
        )
        new_im_handler = self.__im_handler
        derived_update = {}
        for name, new_col in new_columns.items():
            if name in column_names:
                raise ValueError(f"Dataset already has column named {name}")

            if isinstance(new_col, np.ndarray):
                if len(new_col) != len(self.__index):
                    raise ValueError(
                        f"New column {name} has length {len(new_col)} but this dataset "
                        "has length {len(self.__index)}"
                    )
                new_im_handler = new_im_handler.with_new_column(name, new_col)

            elif not new_col.check_parent_existance(column_names):
                raise ValueError(
                    f"Derived column {name} is derived from columns "
                    "that are not in the dataset!"
                )
            else:
                derived_update[name] = new_col
            column_names.add(name)

        new_derived = self.__derived | derived_update
        return DatasetState(
            self.__base_unit_transformations,
            self.__builder,
            self.__index,
            self.__convention,
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

    def __get_im_columns(self, data: table.Table) -> table.Table:
        for colname, column in self.__im_handler.columns():
            data[colname] = column
        return data

    def with_region(self, region: Region):
        """
        Return the same dataset but with a different region
        """
        return DatasetState(
            self.__base_unit_transformations,
            self.__builder,
            self.__index,
            self.__convention,
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

        known_builders = set(self.__builder.columns)
        known_derived = set(self.__derived.keys())
        known_im = set(self.__im_handler.keys())
        unknown_columns = columns - known_builders - known_derived - known_im
        if unknown_columns:
            raise ValueError(
                "Tried to select columns that aren't in this dataset! Missing columns "
                + ", ".join(unknown_columns)
            )

        required_derived = known_derived.intersection(columns)
        required_builders = known_builders.intersection(columns)
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
            required_builders |= additional_columns.intersection(known_builders)
            required_im |= additional_columns.intersection(known_im)
            additional_derived = additional_columns.intersection(known_derived)

        all_required = required_derived | required_builders | required_im

        # Derived columns have to be instantiated in the order they are created in order
        # to ensure chains of derived columns work correctly
        new_derived = {k: v for k, v in self.__derived.items() if k in required_derived}
        # Builders can be performed in any order
        new_builder = self.__builder.with_columns(required_builders)
        new_im_handler = self.__im_handler.with_columns(required_im)

        new_hidden.update(all_required - columns)
        if self.__sort_by is not None and self.__sort_by[0] not in columns:
            new_hidden.add(self.__sort_by[0])

        return DatasetState(
            self.__base_unit_transformations,
            new_builder,
            self.__index,
            self.__convention,
            self.__region,
            self.__header,
            new_im_handler,
            self.__sort_by,
            new_hidden,
            new_derived,
        )

    def sort_by(self, column_name: str, handler: "DatasetHandler", invert: bool):
        return DatasetState(
            self.__base_unit_transformations,
            self.__builder,
            self.__index,
            self.__convention,
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
        self, convention: str, cosmology: Cosmology, redshift: float | table.Column
    ):
        """
        Change the unit convention
        """
        new_transformations = u.get_unit_transition_transformations(
            self.__header.file.unit_convention,
            convention,
            self.__base_unit_transformations,
            cosmology,
            redshift,
        )
        convention_ = u.UnitConvention(convention)
        new_builder = get_table_builder(new_transformations, self.__builder.columns)
        return DatasetState(
            self.__base_unit_transformations,
            new_builder,
            self.__index,
            convention_,
            self.__region,
            self.__header,
            self.__im_handler,
            self.__sort_by,
            self.__hidden,
            self.__derived,
        )
