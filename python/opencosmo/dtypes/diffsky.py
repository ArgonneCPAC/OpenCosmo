from __future__ import annotations

from datetime import datetime  # noqa
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, field_serializer

from opencosmo.column.column import EvaluatedColumn, EvaluateStrategy
from opencosmo.index.ops import reindex_column


class DiffskyVersionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "diffsky_versions"
    diffmah: str
    diffsky: str
    diffstar: str
    diffstarpop: Optional[str] = None
    dsps: str
    jax: str
    numpy: str


class DiffskyCatalogInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "catalog_info"
    README: Optional[str] = None
    mock_version_name: str
    zphot_table: Optional[tuple[float, ...]] = None

    @field_serializer("zphot_table")
    def serialize_zphot_table(self, value):
        if value is not None:
            return list(value)
        return None


def rebuild_top_host_idx(top_host_idx, index):
    result = reindex_column(index, top_host_idx)

    return {"top_host_idx": result}


top_host_idx = EvaluatedColumn(
    rebuild_top_host_idx,
    requires=set(["top_host_idx"]),
    produces=set(["top_host_idx"]),
    format="numpy",
    units={"top_host_idx": None},
    strategy=EvaluateStrategy.VECTORIZE,
    no_cache=True,
)
