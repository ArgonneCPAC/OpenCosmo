from functools import reduce
from typing import TYPE_CHECKING, Iterable

from astropy import table
from astropy.cosmology import Cosmology  # type: ignore

import opencosmo.transformations.units as u
from opencosmo.dataset.column import ColumnBuilder, get_column_builders
from opencosmo.dataset.mask import DerivedColumn
from opencosmo.index import DataIndex
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
        builders: dict[str, ColumnBuilder],
        index: DataIndex,
        convention: u.UnitConvention,
        region: Region,
        hidden: set[str] = set(),
        derived: dict[str, DerivedColumn] = {},
    ):
        self.__base_unit_transformations = base_unit_transformations
        self.__builders = builders
        self.__index = index
        self.__convention = convention
        self.__region = region
        self.__hidden = hidden
        self.__derived: dict[str, DerivedColumn] = derived

    @property
    def index(self):
        return self.__index

    @property
    def builders(self):
        return self.__builders

    @property
    def convention(self):
        return self.__convention

    @property
    def region(self):
        return self.__region

    @property
    def columns(self) -> list[str]:
        columns = set(self.__builders.keys()) | set(self.__derived.keys())
        return list(columns - self.__hidden)

    def get_data(self, handler: "DatasetHandler"):
        """
        Get the data for a given handler.
        """
        data = handler.get_data(builders=self.__builders, index=self.__index)
        data = self.__build_derived_columns(data)
        if self.__hidden:
            data.remove_columns(self.__hidden)
        return data

    def with_index(self, index: DataIndex):
        """
        Return the same dataset state with a new index
        """
        return DatasetState(
            self.__base_unit_transformations,
            self.__builders,
            index,
            self.__convention,
            self.__region,
            self.__hidden,
            self.__derived,
        )

    def with_derived_columns(self, **new_columns: DerivedColumn):
        """
        Add a set of derived columns to the dataset. A derived column is a column that
        has been created based on the values in another column.
        """
        column_names = set(self.builders.keys()) | set(self.__derived.keys())
        for name, new_col in new_columns.items():
            if name in column_names:
                raise ValueError(f"Dataset already has column named {name}")
            elif not new_col.check_parent_existance(column_names):
                raise ValueError(
                    f"Derived column {name} is derived from columns "
                    "that are not in the dataset!"
                )
            column_names.add(name)

        new_derived = self.__derived | new_columns
        return DatasetState(
            self.__base_unit_transformations,
            self.__builders,
            self.__index,
            self.__convention,
            self.__region,
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

    def with_region(self, region: Region):
        """
        Return the same dataset but with a different region
        """
        return DatasetState(
            self.__base_unit_transformations,
            self.__builders,
            self.__index,
            self.__convention,
            region,
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

        known_builders = set(self.__builders.keys())
        known_derived = set(self.__derived.keys())
        unknown_columns = columns - known_builders - known_derived
        if unknown_columns:
            raise ValueError(
                "Tried to select columns that aren't in this dataset! Missing columns "
                + ", ".join(unknown_columns)
            )

        required_derived = known_derived.intersection(columns)
        required_builders = known_builders.intersection(columns)
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
            additional_derived = additional_columns.intersection(known_derived)

        all_required = required_derived | required_builders

        # Derived columns have to be instantiated in the order they are created in order
        # to ensure chains of derived columns work correctly
        new_derived = {k: v for k, v in self.__derived.items() if k in required_derived}
        # Builders can be performed in any order
        new_builders = {b: self.__builders[b] for b in required_builders}

        new_hidden = all_required - columns

        return DatasetState(
            self.__base_unit_transformations,
            new_builders,
            self.__index,
            self.__convention,
            self.__region,
            new_hidden,
            new_derived,
        )

    def take(self, n: int, at: str):
        """
        Take rows from the dataset.
        """
        new_index = self.__index.take(n, at)
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

    def with_units(self, convention: str, cosmology: Cosmology, redshift: float):
        """
        Change the unit convention
        """
        new_transformations = u.get_unit_transition_transformations(
            convention, self.__base_unit_transformations, cosmology, redshift
        )
        convention_ = u.UnitConvention(convention)
        new_builders = get_column_builders(new_transformations, self.__builders.keys())
        return DatasetState(
            self.__base_unit_transformations,
            new_builders,
            self.__index,
            convention_,
            self.__region,
            self.__hidden,
            self.__derived,
        )
