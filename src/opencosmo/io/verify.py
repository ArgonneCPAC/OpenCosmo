from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypedDict

import numpy as np

from opencosmo.index import DataIndex, get_data, get_length

from .schema import FileEntry

if TYPE_CHECKING:
    import h5py
    import numpy as np
    from numpy.typing import DTypeLike

    from opencosmo.index import DataIndex

    from .schema import Schema


""" Columns in a schema are given by their path within the file.
"""


def verify_file(
    schema: Schema,
):
    match schema.type:
        case FileEntry.DATASET:
            return verify_dataset_data(schema)
        case FileEntry.STRUCTURE_COLLECTION:
            return verify_structure_collection_data(schema)

        case _:
            raise ValueError("Unknown file structure!")


def verify_structure_collection_data(schema: Schema):
    if "halo_properties" in schema["children"]:
        link_holder = "halo_properties"
    elif "galaxy_properties" in schema["children"]:
        link_holder = "galaxy_properties"
    else:
        raise ValueError("No valid link holder found in schema!")

    for child_name, child_schema in schema["children"].items():
        if child_name == link_holder:
            links = list(
                filter(lambda cn: "data_linked" in cn, child_schema["columns"].keys())
            )
            raise AttributeError

        match child["entry_type"]:
            case FileEntry.DATASET:
                child_columns = {
                    k: v for k, v in columns.items() if k.startswith(child["path"])
                }
                verify_dataset_data(child_columns)
            case _:
                raise NotImplementedError


def verify_dataset_data(schema: Schema):
    """
    Verify a given dataset is valid. Requiring:
    1. It has a data group
    2. It has a spatial index group
    3. If it has any metadata groups, they are the same length as the data group
    """
    index_root = None
    children = schema.children

    if "data" not in children or "index" not in children:
        raise ValueError("Datasets must have at least a data group and a index group")

    metadata_groups = [
        child
        for name, child in schema.children.items()
        if name not in ["data", "index"] and child.type == FileEntry.COLUMNS
    ]

    verify_column_group(schema.children["data"])
    for child in schema.children["index"].children.values():
        verify_column_group(child)
    for md_child in metadata_groups:
        verify_column_group(md_child)


def verify_column_group(
    schema: Schema,
    verify_root: Optional[str] = None,
    verify_length_by_group=False,
):
    """
    Verify that a given data group is valid. This requires that:
    1. All column writers have the same length
    2. All columns have the same combine strategy
    3. All columns are in the same group

    By default, requires that all columns are in the same group. If "verify_root"
    is set, verifies that all columns are in "verify_root", but does not require
    they are in the same group.

    If verify_length_by_group is set, length verification is done on a per-group basis
    rather than for all columns together. This only has an effect when verify_root
    is set.
    """
    column_names = set()
    group_names = set()
    column_lengths = {}
    column_strategies = set()
    for column_path, column_writer in schema.columns.items():
        try:
            group_name, column_name = column_path.rsplit("/", 1)
        except ValueError:
            group_name = None
            column_name = column_path
        group_names.add(group_name)
        column_names.add(column_name)
        column_lengths[column_path] = len(column_writer)
        column_strategies.add(column_writer.combine_strategy)

    all_column_lengths = set(column_lengths.values())

    if verify_root is None and len(group_names) != 1:
        raise ValueError(
            "Attempted to verify a single column group, but got columns in seperate groups"
        )

    elif verify_root is not None and not all(
        gn.startswith(verify_root) for gn in group_names
    ):
        raise ValueError(f"Columns in this group should be relative to {verify_root}")

    if len(all_column_lengths) != 1 and not verify_length_by_group:
        raise ValueError(
            "Columns within a single group should always have the same length!"
        )
    elif verify_length_by_group:
        verify_lengths_by_groups(group_names, column_lengths)

    if len(column_strategies) != 1:
        raise ValueError(
            "Columns within a single group should always have the same combine strategy!"
        )

    return (group_names.pop(), column_lengths, column_strategies.pop())


def verify_lengths_by_groups(groups: set[str], column_lengths: dict[str, int]):
    for group_name in groups:
        columns_in_group = filter(
            lambda kv: kv[0].startswith(group_name), column_lengths.items()
        )
        lengths = set(map(lambda kv: kv[1], columns_in_group))
        if len(lengths) != 1:
            raise ValueError(
                f"Columns in group {group_name} do not have the same length!"
            )
