from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import FileEntry

if TYPE_CHECKING:
    from .schema import Schema


""" 
Verification is just a check to make sure that the data going into the file is valid. It is intentionally centralized,
such that if you create a new data type and implement "make_schema" it will fail at this step until you add verification.
"""


class ZeroLengthError(Exception):
    pass


def verify_file(
    schema: Schema,
):
    match schema.type:
        case FileEntry.DATASET:
            return verify_dataset_data(schema)
        case FileEntry.STRUCTURE_COLLECTION:
            return verify_structure_collection_data(schema)
        case FileEntry.LIGHTCONE:
            verify_lightcone_collection_schema(schema)
        case FileEntry.SIMULATION_COLLECTION:
            for name, ds_schema in schema.children.items():
                match ds_schema.type:
                    case FileEntry.DATASET:
                        verify_dataset_data(ds_schema)
                    case FileEntry.STRUCTURE_COLLECTION:
                        verify_structure_collection_data(ds_schema)
        case FileEntry.HEALPIX_MAP:
            verify_dataset_data(schema, has_index=False)
        case _:
            raise ValueError("Unknown file structure!")


def verify_column_group(schema: Schema, require_data: bool = False):
    """
    Verify that a given data group is valid. This requires that:
    1. All column writers have the same length
    2. All columns have the same combine strategy
    3. All columns are in the same group
    """
    column_names = set()
    group_names = set()
    column_lengths = {}
    column_strategies = set()
    for column_path, column_writer in schema.columns.items():
        try:
            group_name, column_name = column_path.rsplit("/", 1)
        except ValueError:
            group_name = ""
            column_name = column_path
        group_names.add(group_name)
        column_names.add(column_name)
        column_lengths[column_path] = len(column_writer)
        column_strategies.add(column_writer.combine_strategy)

    all_column_lengths = set(column_lengths.values())

    if len(all_column_lengths) != 1:
        raise ValueError(
            "Columns within a single group should always have the same length!"
        )
    elif (group_length := all_column_lengths.pop()) == 0 and require_data:
        raise ZeroLengthError

    if len(column_strategies) != 1:
        raise ValueError(
            "Columns within a single group should always have the same combine strategy!"
        )

    return (group_names.pop(), group_length, column_strategies.pop())


def verify_dataset_data(schema: Schema, has_index=True):
    """
    Verify a given dataset is valid. Requiring:
    1. It has a data group
    2. It has a spatial index group (if has_index = True)
    3. If it has any metadata groups, they are the same length as the data group

    Once this is verified, we delegate to verify_column_group to ensure
    individual column groups are valid.
    """
    children = schema.children

    if "data" not in children or ("index" not in children and has_index):
        raise ValueError("Datasets must have at least a data group and a index group")

    metadata_groups = [
        child
        for name, child in schema.children.items()
        if name not in ["data", "index"] and child.type == FileEntry.COLUMNS
    ]

    _, data_length, data_combine_strategy = verify_column_group(
        schema.children["data"], require_data=True
    )
    if has_index:
        for child in schema.children["index"].children.values():
            verify_column_group(child)
    for md_child in metadata_groups:
        _, md_length, md_combine_strategy = verify_column_group(md_child)
        if md_length != data_length or md_combine_strategy != data_combine_strategy:
            raise ValueError(
                "Metadata groups must be the same length and have the same combine strategy as data groups!"
            )


def verify_lightcone_collection_schema(schema: Schema):
    """
    Verify a lightcone collection. Note that a single dataset
    can also technically be a lighcone collection, if is_lightcone is
    set to true in its header. Mostly just delegates to underlying
    dataset checks.
    """
    if len(schema.children) < 1:
        raise ValueError("Expect at least one lightcone child!")
    elif "data" in schema.children:
        # Single-dataset lightcone
        return verify_dataset_data(schema)
    for key, child_schema in schema.children.items():
        verify_dataset_data(child_schema)


def verify_structure_collection_data(schema: Schema):
    """
    Structure collections have a lot going on, but they are mostly just datasets. The only
    thing we have to check is that we have an explicit "data_linked" group in any link
    holders, and we then verify the individual dataset.

    """
    if "halo_properties" in schema.children:
        link_holder = "halo_properties"
    elif "galaxy_properties" in schema.children:
        link_holder = "galaxy_properties"
    else:
        raise ValueError("No valid link holder found in schema!")

    for child_name, child_schema in schema.children.items():
        if child_name == link_holder:
            has_link = any(
                map(lambda cn: "data_linked" in cn, child_schema.children.keys())
            )
            if not has_link:
                raise ValueError(
                    f'Source dataset {child_name} does not have expected "data_linked" group'
                )

        match child_schema.type:
            case FileEntry.DATASET:
                verify_dataset_data(child_schema)
            case FileEntry.STRUCTURE_COLLECTION:
                verify_structure_collection_data(child_schema)
            case _:
                raise ValueError("Got an unknown child for structure collection!")
