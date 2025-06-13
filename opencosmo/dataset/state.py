from typing import Iterable

from astropy import table
from astropy.cosmology import Cosmology  # type: ignore

import opencosmo.transformations.units as u
from opencosmo.dataset.column import ColumnBuilder, get_column_builders
from opencosmo.dataset.mask import DerivedColumn
from opencosmo.index import DataIndex
from opencosmo.spatial.protocols import Region


class DatasetState:
    """
    Holds any mutable state required by the dataset. Anything that remains unchanged
    will be held by the datases itself
    """

    def __init__(
        self,
        base_unit_transformations: dict,
        builders: dict[str, ColumnBuilder],
        index: DataIndex,
        convention: u.UnitConvention,
        region: Region,
        derived: dict[str, DerivedColumn] = {},
    ):
        self.__base_unit_transformations = base_unit_transformations
        self.__builders = builders
        self.__index = index
        self.__convention = convention
        self.__region = region
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
        return list(self.__builders.keys()) + list(self.__derived.keys())

    def with_index(self, index: DataIndex):
        return DatasetState(
            self.__base_unit_transformations,
            self.__builders,
            index,
            self.__convention,
            self.__region,
        )

    def with_derived_columns(self, **new_columns: DerivedColumn):
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
            new_derived,
        )

    def build_derived_columns(self, data: table.Table) -> table.Table:
        for colname, column in self.__derived.items():
            new_column = column.evaluate(data)
            data[colname] = new_column
        return data

    def with_builders(self, builders: dict[str, ColumnBuilder]):
        return DatasetState(
            self.__base_unit_transformations,
            builders,
            self.__index,
            self.__convention,
            self.__region,
        )

    def with_region(self, region: Region):
        return DatasetState(
            self.__base_unit_transformations,
            self.__builders,
            self.__index,
            self.__convention,
            region,
        )

    def select(self, columns: str | Iterable[str]):
        if isinstance(columns, str):
            columns = [columns]
        columns = [str(col) for col in columns]

        try:
            new_builders = {col: self.__builders[col] for col in columns}
        except KeyError:
            known_columns = set(self.__builders.keys())
            unknown_columns = set(columns) - known_columns
            raise ValueError(
                "Tried to select columns that aren't in this dataset! Missing columns "
                + ", ".join(unknown_columns)
            )
        return DatasetState(
            self.__base_unit_transformations,
            new_builders,
            self.__index,
            self.__convention,
            self.__region,
        )

    def take(self, n: int, at: str):
        new_index = self.__index.take(n, at)
        return self.with_index(new_index)

    def take_range(self, start: int, end: int):
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
        )
