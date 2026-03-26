from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Iterable, NamedTuple

if TYPE_CHECKING:
    from opencosmo.column.column import ConstructedColumn
    from opencosmo.index import DataIndex


class QueryOp:
    pass


class DatasetView(NamedTuple):
    names: set[str]
    producers: list[ConstructedColumn]
    index: DataIndex
    operations: list[QueryOp]


class TakeOp(NamedTuple):
    n: int
    strategy: TakeStrategy


class TakeStrategy(Enum):
    START = "start"
    END = "end"
    RANDOM = "random"


def rebuild_view(view: DatasetView, **updates):
    new_view = {
        "names": view.names,
        "producers": view.producers,
        "index": view.index,
        "operations": view.operations,
    } | updates
    return DatasetView(**new_view)


def select_names(view: DatasetView, names: Iterable[str]) -> DatasetView:
    new_names = set(names)
    if missing := view.names.difference(new_names):
        raise ValueError(f"Dataset does not contain columns {missing}")
    return rebuild_view(view, names=new_names)


def append_op(view: DatasetView, op: QueryOp) -> DatasetView:
    new_operations = view.operations + [op]
    return rebuild_view(view, operations=new_operations)


def add_column(view: DatasetView, column: ConstructedColumn) -> DatasetView:
    added_names = column.produces
    if overlaps := view.names.intersection(added_names):
        raise ValueError(f"Dataset already contains columns {overlaps}")

    new_names = view.names.union(added_names)
    new_producers = view.producers + [column]
    return rebuild_view(view, names=new_names, producers=new_producers)


take_rows = append_op
apply_filter = append_op
