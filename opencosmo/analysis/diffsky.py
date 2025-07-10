from collections import namedtuple
from typing import NamedTuple, Type, TypeVar

from opencosmo import Dataset

DIFFMAH_INPUT = namedtuple(
    "DIFFMAH_INPUT", ["logm0", "logtc", "early_index", "late_index", "t_peak"]
)

T = TypeVar("T", bound=NamedTuple)


def make_named_tuple(dataset: Dataset, input_tuple: Type[T]) -> T:
    required_columns = input_tuple._fields
    data = dataset.select(required_columns).data
    output = {c: data[c].value for c in required_columns}
    return input_tuple(**output)  # type: ignore
