from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple, TypeVar

if TYPE_CHECKING:
    from opencosmo import Dataset

DIFFMAH_INPUT = namedtuple(
    "DIFFMAH_INPUT", ["logm0", "logtc", "early_index", "late_index", "t_peak"]
)

T = TypeVar("T", bound=NamedTuple)


def make_named_tuple[T: NamedTuple](dataset: Dataset, input_tuple: type[T]) -> T:
    required_columns = input_tuple._fields
    data = dataset.select(required_columns).data
    output = {c: data[c].value for c in required_columns}
    return input_tuple(**output)  # type: ignore
