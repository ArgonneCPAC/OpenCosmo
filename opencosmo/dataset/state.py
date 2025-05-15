from typing import Iterable

from astropy.cosmology import Cosmology

import opencosmo.transformations.units as u
from opencosmo.dataset.column import ColumnBuilder, get_column_builders
from opencosmo.dataset.index import DataIndex


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
    ):
        self.__base_unit_transformations = base_unit_transformations
        self.__builders = builders
        self.__index = index
        self.__convention = convention

    @property
    def index(self):
        return self.__index

    @property
    def builders(self):
        return self.__builders

    @property
    def convention(self):
        return self.__convention

    def with_index(self, index: DataIndex):
        return DatasetState(
            self.__base_unit_transformations, self.__builders, index, self.__convention
        )

    def with_builders(self, builders: dict[str, ColumnBuilder]):
        return DatasetState(
            self.__base_unit_transformations, builders, self.__index, self.__convention
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
            self.__base_unit_transformations, new_builders, self.__index, convention_
        )
