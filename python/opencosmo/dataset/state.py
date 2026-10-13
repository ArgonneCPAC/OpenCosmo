from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from weakref import finalize

import astropy.units as u

from opencosmo.column.cache import ColumnCache
from opencosmo.column.column import RawColumn
from opencosmo.handler.empty import EmptyHandler
from opencosmo.handler.hdf5 import Hdf5Handler
from opencosmo.handler.memory import InMemoryHandler
from opencosmo.index import from_size
from opencosmo.units import UnitConvention
from opencosmo.units.handler import (
    make_unit_handler_from_unit_strings,
    make_unit_handler_from_units,
)
from opencosmo.uuid import (
    get_in_memory_dataset_uuid,
    get_raw_column_uuid,
)

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np

    from opencosmo.column.column import ConstructedColumn
    from opencosmo.handler.protocols import DataCache, DataHandler
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.spatial.protocols import Region, SpatialIndex
    from opencosmo.units.handler import UnitHandler


def deregister_state(id: int, cache: DataCache):
    cache.deregister_column_group(id)


@dataclass(frozen=True)
class DatasetState:
    """
    Main state container for the Dataset. Functions for manipulating it can be found below. The dataclass
    itself only exposes basic lookup operations.
    """

    uuid: UUID
    producers: dict[UUID, ConstructedColumn]
    raw_data_handler: DataHandler
    cache: DataCache
    unit_handler: UnitHandler
    header: OpenCosmoHeader
    spatial_index: SpatialIndex | None
    region: Region | None
    column_map: dict[str, UUID]
    open_kwargs: dict[str, Any]
    sort_key: tuple[str, bool, bool] | None

    def __post_init__(self):
        self.cache.register_column_group(id(self), self.column_map)
        finalize(self, deregister_state, id(self), self.cache)

    @property
    def columns(self) -> list[str]:
        sort_to_drop: int | str = -1
        if self.sort_key is not None and self.sort_key[2]:
            sort_to_drop = self.sort_key[0]

        return [c for c in self.column_map if c != sort_to_drop]

    @property
    def descriptions(self):
        all_descriptions = {}
        for producer in self.producers.values():
            update = {name: producer.description for name in producer.produces}
            all_descriptions |= update
        all_descriptions |= self.cache.descriptions

        return {
            name: description
            for name, description in all_descriptions.items()
            if name in self.columns
        }

    @property
    def kwargs(self):
        return self.open_kwargs

    @property
    def raw_index(self):
        from opencosmo.dataset.operations import get_sorted_index
        from opencosmo.index.mask import into_array

        if (si := get_sorted_index(self)) is not None:
            ni = into_array(self.raw_data_handler.index)
            return ni[si]
        return self.raw_data_handler.index

    @property
    def units(self):
        units = self.unit_handler.current_units
        return {name: units[name] for name in self.columns}

    @property
    def convention(self):
        return self.unit_handler.current_convention

    def __len__(state: DatasetState) -> int:
        if isinstance(state.raw_data_handler, EmptyHandler):
            return len(state.cache)
        return len(state.raw_data_handler)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Factory functions (replace classmethods)
# ---------------------------------------------------------------------------


def state_from_target(
    target: DatasetTarget,
    unit_convention: UnitConvention,
    open_kwargs: dict[str, Any],
    index: DataIndex | None = None,
    spatial_index: SpatialIndex | None = None,
    region: Region | None = None,
) -> DatasetState:
    handler = Hdf5Handler(
        target.data_group,
        {cn: None for cn in target.column_names},
        index if index is not None else from_size(target.row_count),
        target.load_conditions,
        descriptions=target.column_descriptions,
    )
    unit_handler = make_unit_handler_from_unit_strings(
        target.column_units, target.header, unit_convention
    )
    descriptions = target.column_descriptions
    uuids = target.column_uuids

    raw_producers = [
        RawColumn(
            cname,
            descriptions.get(cname, "None"),
            _uuid=uuid,
            on_disk=True,
        )
        for cname, uuid in uuids.items()
    ]
    column_map = {p.name: p.uuid for p in raw_producers}
    producers: dict[UUID, ConstructedColumn] = {p.uuid: p for p in raw_producers}
    cache = ColumnCache.empty()
    return DatasetState(
        uuid=target.uuid,
        producers=producers,
        raw_data_handler=handler,
        cache=cache,
        unit_handler=unit_handler,
        header=target.header,
        spatial_index=spatial_index,
        region=region,
        column_map=column_map,
        open_kwargs=open_kwargs,
        sort_key=None,
    )


def state_in_memory(
    data_columns: dict,
    header: OpenCosmoHeader,
    unit_convention: UnitConvention,
    open_kwargs: dict[str, Any],
    descriptions: dict[str, str] | None = None,
    index: DataIndex | None = None,
    spatial_index: SpatialIndex | None = None,
    region: Region | None = None,
) -> DatasetState:
    descriptions = descriptions or {}

    all_columns = dict(data_columns)
    raw_producers = [
        RawColumn(
            cname, descriptions.get(cname, "None"), get_raw_column_uuid(cname, set())
        )
        for cname in all_columns
    ]
    column_map = {p.name: p.uuid for p in raw_producers}
    producers: dict[UUID, ConstructedColumn] = {p.uuid: p for p in raw_producers}

    cache = ColumnCache.empty()
    if all_columns:
        uuid_data = {p.uuid: {p.name: all_columns[p.name]} for p in raw_producers}
        cache.add_data(uuid_data, descriptions)

    units: dict[str, u.Unit] = {}
    for name, column in all_columns.items():
        units[name] = None
        if isinstance(column, u.Quantity):
            units[name] = column.unit

    unit_handler = make_unit_handler_from_units(units, header, unit_convention)

    return DatasetState(
        uuid=get_in_memory_dataset_uuid(data_columns),
        producers=producers,
        raw_data_handler=EmptyHandler(),
        cache=cache,
        unit_handler=unit_handler,
        header=header,
        spatial_index=spatial_index,
        region=region,
        column_map=column_map,
        open_kwargs=open_kwargs,
        sort_key=None,
    )


def state_from_evaluated_data(
    data_columns: dict[str, np.ndarray | u.Quantity],
    header: OpenCosmoHeader,
    descriptions: dict[str, str] | None = None,
) -> DatasetState:
    """Build a dataset from evaluated values in the file's unit convention."""
    descriptions = {
        name: (descriptions or {}).get(name, "None") for name in data_columns
    }
    all_known_uuids: set[UUID] = set()
    raw_producers = []
    for name in data_columns:
        producer = RawColumn(
            name,
            descriptions[name],
            get_raw_column_uuid(name, all_known_uuids),
            no_cache=True,
            on_disk=True,
        )
        raw_producers.append(producer)
        all_known_uuids.add(producer.uuid)
    column_map = {producer.name: producer.uuid for producer in raw_producers}
    producers: dict[UUID, ConstructedColumn] = {
        producer.uuid: producer for producer in raw_producers
    }
    units = {
        name: column.unit if isinstance(column, u.Quantity) else None
        for name, column in data_columns.items()
    }
    raw_data = {
        name: column.value if isinstance(column, u.Quantity) else column
        for name, column in data_columns.items()
    }
    length = len(next(iter(raw_data.values()))) if raw_data else 0
    handler = InMemoryHandler(raw_data, units, descriptions, from_size(length))
    unit_convention = UnitConvention(header.file.unit_convention)

    return DatasetState(
        uuid=get_in_memory_dataset_uuid(data_columns),
        producers=producers,
        raw_data_handler=handler,
        cache=ColumnCache.empty(),
        unit_handler=make_unit_handler_from_units(
            units, header, target_convention=unit_convention
        ),
        header=header.with_units(unit_convention),
        spatial_index=None,
        region=None,
        column_map=column_map,
        open_kwargs={},
        sort_key=None,
    )


def exit_state(state: DatasetState, *exec_details):
    return None
