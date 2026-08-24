from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
from uuid import UUID, uuid5

if TYPE_CHECKING:
    import h5py

NAMESPACE = UUID("02af60eb-68cc-43f2-817b-eb0967ebfde4")


def get_hdf5_column_uuid(column: h5py.Dataset) -> UUID:
    file_name = column.file.filename
    path = column.name

    id = f"{file_name}::{path}"

    return uuid5(NAMESPACE, id)


def get_derived_column_uuid(
    dep_uuids: Iterable[UUID], output_names: Iterable[str], all_known_uuids: set[UUID]
):
    dep_uuids_str = [str(uuid) for uuid in sorted(dep_uuids)]
    output_names = sorted(output_names)
    id = f"{'+'.join(dep_uuids_str)}->{'+'.join(output_names)}"
    uuid = uuid5(NAMESPACE, id)
    return _get_free(uuid, all_known_uuids)


def get_raw_column_uuid(output_name: str, all_known_uuids: set[UUID]):
    id = f"raw::{output_name}"

    uuid = uuid5(NAMESPACE, id)
    return _get_free(uuid, all_known_uuids)


def _get_free(uuid: UUID, all_known_uuids: set[UUID]):
    while uuid in all_known_uuids:
        uuid = uuid5(NAMESPACE, str(uuid))
    return uuid
