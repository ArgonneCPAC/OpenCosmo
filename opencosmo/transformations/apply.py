import numpy as np
from astropy.table import Table  # type: ignore

from opencosmo.transformations import protocols as t

"""
Routines for applying transformations tables. The table that is
passed into these routines is always a copy of the raw data, so 
it is safe to modify in-place. However they should still return
the updated versions at the end.
"""


def apply_column_transformations(
    table: Table, transformations: list[t.ColumnTransformation]
):
    """
    Apply a list of column transformations to a table. If multiple
    transformations are present for the same column, they will simply
    be applied in the order they appear in the list.
    """
    for tr in transformations:
        column_name = tr.column_name
        if column_name not in table.columns:
            raise ValueError(f"Column {column_name} not found in table")
        column = table[column_name]
        if (new_column := tr(column)) is not None:
            table[column_name] = new_column
    return table


def apply_all_columns_transformations(
    table: Table, transformations: list[t.ColumnTransformation]
):
    """
    Apply a list of column transformations to all columns in the table.
    This is useful for transformations that should be applied to all columns,
    such as unit conversions.
    """
    for tr in transformations:
        for column_name in table.columns:
            column = table[column_name]
            if (new_column := tr(column)) is not None:
                table[column_name] = new_column
    return table


def apply_table_transformations(
    table: Table, transformations: list[t.TableTransformation]
):
    """
    Apply transformations to the table as a whole. These transformations
    are applied after individual column transformations.
    """
    output_table = table
    for tr in transformations:
        if (new_table := tr(output_table)) is not None:
            output_table = new_table
    return output_table


def apply_filter_transformations(
    table: Table, transformations: list[t.FilterTransformation]
):
    """
    Filter transformations produce boolean masks that are applied to the table
    to remove rows.
    """
    mask = np.ones(len(table), dtype=bool)
    for tr in transformations:
        if (new_mask := tr(table)) is not None:
            mask &= new_mask
    return table[mask]
