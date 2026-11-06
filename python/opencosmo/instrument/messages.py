"""Log records for operations that cannot be replayed from a message."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from opencosmo.serde import DatasetWithNewColumnsMessage


class DatasetEvaluateMessage(BaseModel):
    """A record of an inserting call to :meth:`opencosmo.Dataset.evaluate`.

    The evaluated function cannot be serialized, so only the names of the
    columns it consumed and produced are recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["evaluate"] = "evaluate"
    method: ClassVar[str] = "evaluate"
    requires: tuple[str, ...]
    produces: tuple[str, ...]


class DatasetWithDataColumnsMessage(BaseModel):
    """A record of a :meth:`opencosmo.Dataset.with_new_columns` call with in-memory data.

    Expression columns are recorded in ``call``. Array and quantity values are
    not serialized, so only their names are recorded in ``data_columns``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["with_data_columns"] = "with_data_columns"
    method: ClassVar[str] = "with_new_columns"
    call: DatasetWithNewColumnsMessage
    data_columns: tuple[str, ...]
