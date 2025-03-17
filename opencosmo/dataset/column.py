from __future__ import annotations

from collections.abc import Iterable, Sequence

from astropy.table import Column  # type: ignore

import opencosmo.transformations as t


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


class ColumnBuilder:
    """
    OpenCosmo operates on columns of data, only producing an actual full Astropy table
    when data is actually requested. Things like filtering, selecting, and changing
    units are repesented as transformations on the given column.

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
            transformed_column = transformation(new_column)
            if transformed_column is not None:
                new_column = transformed_column
        return new_column
