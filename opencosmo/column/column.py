from __future__ import annotations

from collections.abc import Iterable, Sequence

from astropy.table import Column  # type: ignore
import astropy.units as u # type: ignore
import operator as op

import opencosmo.transformations as t
from opencosmo.column.filter import Filter
from opencosmo.column.derived import DerivedColumn


def col(column_name: str) -> Column:
    return OcColumn(column_name)


def get_column_builders(
    transformations: t.TransformationDict, column_names: Iterable[str]
) -> dict[str, ColumnBuilder]:
    """
    This function creates a dictionary of ColumnBuilders from a dictionary of
    transformations. The keys of the dictionary are the column names and the
    values are the ColumnBuilders.
    """
    column_transformations = transformations.get(t.TransformationType.COLUMN, [])
    all_column_transformations = transformations.get(
        t.TransformationType.ALL_COLUMNS, []
    )
    if not all(
        isinstance(transformation, t.AllColumnTransformation)
        for transformation in all_column_transformations
    ):
        raise ValueError("Expected AllColumnTransformation.")
    column_builders: dict[str, list[t.Transformation]] = {
        name: [] for name in column_names
    }
    for transformation in column_transformations:
        if not hasattr(transformation, "column_name"):
            raise ValueError(
                f"Expected ColumnTransformation, got {type(transformation)}."
            )
        if transformation.column_name not in column_builders:
            continue
        column_builders[transformation.column_name].append(transformation)

    for column_name in column_names:
        column_builders[column_name].extend(all_column_transformations)
    return {
        name: ColumnBuilder(name, builders)
        for name, builders in column_builders.items()
    }


class OcColumn:
    """
    A column representa a column in the table. This is used first and foremost
    for filtering purposes. For example, if a user has loaded a dataset they
    can filter it with

    dataset.filter(oc.Col("column_name") < 5)

    In practice, this is just a factory class that returns filter
    """

    def __init__(self, column_name: str):
        self.column_name = column_name

    # mypy doesn't reason about eq and neq correctly
    def __eq__(self, other: float | u.Quantity) -> Filter:  # type: ignore
        return Filter(self.column_name, other, op.eq)

    def __ne__(self, other: float | u.Quantity) -> Filter:  # type: ignore
        return Filter(self.column_name, other, op.ne)

    def __gt__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.gt)

    def __ge__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.ge)

    def __lt__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.lt)

    def __le__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.le)

    def __add__(self, other: Column) -> DerivedColumn:
        return DerivedColumn([self.column_name, other.column_name], [op.add])

    def __sub__(self, other: Column) -> DerivedColumn:
        return DerivedColumn([self.column_name, other.column_name], [op.sub])

    def __mul__(self, other: Column) -> DerivedColumn:
        return DerivedColumn([self.column_name, other.column_name], [op.mul])

    def __truediv__(self, other: Column) -> DerivedColumn:
        return DerivedColumn([self.column_name, other.column_name], [op.truediv])


class ColumnBuilder:
    """
    OpenCosmo operates on columns of data, only producing an actual full Astropy table
    when data is actually requested. Things like filtering, selecting, and changing
    units are repesented as transformations on the given column.

    Column builders only operate on columns that are present in the source data. 
    Columns that are derived used the DerivedColumn class in dataset/derived.py

    The handler is responsible for actually getting the data from the source and
    feeding it to the ColumBuilder.

    """

    def __init__(
        self,
        name: str,
        transformations: Sequence[t.Transformation],
    ):
        self.column_name = name
        self.transformations = transformations

    def build(self, data: Column):
        """
        The column should always come to the builder without
        units.
        """
        if data.unit is not None:
            raise ValueError("Data should not have units when building a column.")

        new_column = data
        for transformation in self.transformations:
            new_column = transformation(new_column)
            if new_column is None:
                new_column = data
                continue
        return new_column
