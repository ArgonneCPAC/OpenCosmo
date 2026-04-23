from __future__ import annotations

from datetime import datetime  # noqa
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer

from opencosmo.column.column import EvaluatedColumn, EvaluateStrategy
from opencosmo.index.ops import reindex_column
from opencosmo.plugins.plugin import PluginType, register_plugin

if TYPE_CHECKING:
    from opencosmo import Dataset, Lightcone
    from opencosmo.dataset.state import DatasetState


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


def __offset(top_host_idx, offset):
    output = top_host_idx
    output[output >= 0] += offset
    return {"top_host_idx": output}


def offset_top_host_idx(datasets: list[Dataset]):
    lengths = [len(ds) for ds in datasets]
    offsets = np.cumsum(lengths)
    output_datasets = [datasets[0]]
    for offset, ds in zip(offsets, datasets[1:]):
        output_ds = ds.evaluate(
            __offset,
            offset=offset,
            vectorize=True,
            allow_overwrite=True,
        )
        output_datasets.append(output_ds)  # type: ignore
    return output_datasets


def rebuild_top_host_idx(top_host_idx, index):
    result = reindex_column(index, top_host_idx)

    return {"top_host_idx": result}


def top_host_idx_plugin(dataset: DatasetState):
    top_host_idx = EvaluatedColumn(
        rebuild_top_host_idx,
        requires=set(["top_host_idx"]),
        produces=set(["top_host_idx"]),
        format="numpy",
        units={"top_host_idx": None},
        strategy=EvaluateStrategy.VECTORIZE,
        no_cache=True,
    )
    return dataset.with_new_columns(updated_host_idx=top_host_idx, allow_overwrite=True)


def top_host_idx_offset_plugin(lightcone: Lightcone) -> dict[str, Dataset]:
    cs = 0
    output = {}

    def top_host_idx(top_host_idx, offset):
        top_host_idx[top_host_idx >= 0] += offset
        return top_host_idx

    for key, ds in lightcone.items():
        output[key] = ds.evaluate(
            top_host_idx, allow_overwrite=True, vectorize=True, offset=cs
        )
        cs += len(ds)

    return output


def top_host_idx_verifier[T: (DatasetState, Dataset, Lightcone)](dataset: T) -> bool:
    return (
        dataset.header.file.data_type == "synthetic_galaxies"
        and "top_host_idx" in dataset.columns
    )


register_plugin(PluginType.DatasetOpen, top_host_idx_verifier, top_host_idx_plugin)

register_plugin(  # type: ignore
    PluginType.LightconeInstantiate, top_host_idx_verifier, top_host_idx_offset_plugin
)
