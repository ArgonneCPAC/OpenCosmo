from types import SimpleNamespace
from uuid import UUID, uuid5

import astropy.units as u
import numpy as np
import pytest
from opencosmo.dtypes.file import DatasetType
from opencosmo.serde import (
    CollectionMember,
    ColumnSummary,
    DatasetSummary,
    HealpixMapSummary,
    LightconeSummary,
    ResultSummary,
    SerdeError,
    SimulationCollectionSummary,
    StructureCollectionSummary,
    serialize_result,
)
from pydantic import TypeAdapter, ValidationError

from opencosmo.collection import (
    HealpixMap,
    Lightcone,
    SimulationCollection,
    StructureCollection,
)
from opencosmo.dataset import Dataset
from opencosmo.spatial import HealpixRegion
from opencosmo.units import UnitConvention


class StubDataset(Dataset):
    def __init__(self, length: int = 3):
        self.length = length
        self.test_header = SimpleNamespace(
            file=SimpleNamespace(
                origin="HACC",
                data_type="halo_properties",
                is_lightcone=False,
                redshift=0.5,
                step=42,
                unit_convention="comoving",
            )
        )

    def __len__(self):
        return self.length

    @property
    def columns(self):
        return ["mass", "tag"]

    @property
    def descriptions(self):
        return {"mass": "Particle mass", "tag": None}

    @property
    def units(self):
        return {"mass": u.Msun, "tag": None}

    @property
    def header(self):
        return self.test_header

    @property
    def _state(self):
        return SimpleNamespace(convention=UnitConvention.PHYSICAL)

    @property
    def sorted_by(self):
        return "tag"

    @property
    def region(self):
        return None


class StubSimulationCollection(SimulationCollection):
    def __init__(self, children):
        self.children = children

    def __len__(self):
        return len(self.children)

    def items(self):
        return self.children.items()


class StubHealpixMap(HealpixMap):
    def __init__(self, child):
        dict.__init__(self, {"density": child})

    def __len__(self):
        return len(next(iter(self.values())))

    @property
    def columns(self):
        return next(iter(self.values())).columns

    @property
    def descriptions(self):
        return next(iter(self.values())).descriptions

    @property
    def header(self):
        return next(iter(self.values())).header

    @property
    def region(self):
        return HealpixRegion(np.array([1, 4, 9], dtype=np.int64), 8)

    @property
    def z_range(self):
        return (0.1, 0.9)

    @property
    def nside(self):
        return 8

    @property
    def nside_lr(self):
        return 4

    @property
    def ordering(self):
        return "NESTED"

    @property
    def full_sky(self):
        return False

    @property
    def pixels(self):
        return np.array([1, 4, 9], dtype=np.int64)


class StubLightcone(Lightcone):
    def __init__(self, child, map_value=None):
        dict.__init__(self, {0.5: child})
        self.map_value = map_value

    @property
    def columns(self):
        return next(iter(self.values())).columns

    @property
    def descriptions(self):
        return next(iter(self.values())).descriptions

    @property
    def units(self):
        return next(iter(self.values())).units

    @property
    def header(self):
        return next(iter(self.values())).header

    @property
    def sorted_by(self):
        return None

    @property
    def region(self):
        return None

    @property
    def z_range(self):
        return (0.4, 0.6)

    @property
    def map(self):
        return self.map_value


class StubStructureCollection(StructureCollection):
    def __init__(self, source, linked):
        self.source = source
        self.linked = linked

    def __len__(self):
        return len(self.source)

    @property
    def dtype(self):
        return "halo_properties"

    @property
    def header(self):
        return self.source.header

    @property
    def region(self):
        return self.source.region

    @property
    def sorted_by(self):
        return self.source.sorted_by

    def items(self):
        return iter((("halo_particles", self.linked), ("halo_properties", self.source)))


def test_summary_models_forbid_extra_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ColumnSummary(name="mass", description=None, unit=None, unexpected=True)


def test_result_summary_validates_recursive_discriminator():
    summary = TypeAdapter(ResultSummary).validate_python(
        {
            "kind": "simulation_collection",
            "uuid": None,
            "length": 1,
            "members": [
                {
                    "key": "run-a",
                    "value": {
                        "kind": "dataset",
                        "uuid": None,
                        "length": 0,
                        "columns": [],
                        "header": {
                            "origin": "HACC",
                            "data_type": "halo_properties",
                            "is_lightcone": False,
                            "redshift": None,
                            "step": None,
                            "file_unit_convention": "comoving",
                        },
                        "current_unit_convention": "physical",
                        "sorted_by": None,
                        "region": None,
                    },
                }
            ],
        }
    )

    assert isinstance(summary, SimulationCollectionSummary)
    assert isinstance(summary.members[0].value, DatasetSummary)


def test_dataset_serialization_preserves_complete_column_metadata():
    runtime_uuid = UUID("f5e01879-4d54-49ed-a840-0fe5f3e330cb")
    summary = serialize_result(StubDataset(), resolve_uuid=lambda _: runtime_uuid)

    assert isinstance(summary, DatasetSummary)
    assert summary.uuid == runtime_uuid
    assert summary.columns == (
        ColumnSummary(name="mass", description="Particle mass", unit="solMass"),
        ColumnSummary(name="tag", description=None, unit=None),
    )
    assert summary.current_unit_convention is UnitConvention.PHYSICAL
    assert summary.header.data_type is DatasetType.halo_properties
    assert summary.model_dump(mode="json")["uuid"] == str(runtime_uuid)


def test_healpix_serialization_never_includes_pixels():
    summary = serialize_result(StubHealpixMap(StubDataset()))

    assert isinstance(summary, HealpixMapSummary)
    assert summary.healpix.pixel_count == 3
    assert summary.region is not None
    assert summary.region.kind == "healpix"
    dumped = summary.model_dump(mode="json")
    assert "pixels" not in str(dumped)


def test_lightcone_serialization_includes_members_and_map():
    child = StubDataset()
    summary = serialize_result(StubLightcone(child, StubHealpixMap(child)))

    assert isinstance(summary, LightconeSummary)
    assert summary.redshift_range == (0.4, 0.6)
    assert summary.members[0].key == "0.5"
    assert isinstance(summary.map, HealpixMapSummary)


def test_structure_source_is_serialized_as_regular_member():
    summary = serialize_result(StubStructureCollection(StubDataset(), StubDataset(7)))

    assert isinstance(summary, StructureCollectionSummary)
    assert [member.key for member in summary.members] == [
        "halo_particles",
        "halo_properties",
    ]
    assert summary.members[1].value.length == summary.length


def test_uuid_resolver_is_applied_recursively():
    child = StubDataset()
    collection = StubSimulationCollection({"run-a": child})

    def resolve_uuid(value):
        return uuid5(UUID(int=0), type(value).__name__)

    summary = serialize_result(collection, resolve_uuid=resolve_uuid)

    assert summary.uuid == resolve_uuid(collection)
    assert summary.members[0].key == "run-a"
    assert summary.members[0].value.uuid == resolve_uuid(child)


def test_collection_member_rejects_unknown_result_kind():
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        CollectionMember.model_validate(
            {"key": "bad", "value": {"kind": "unknown", "uuid": None}}
        )


def test_serialize_result_rejects_unsupported_types():
    result = serialize_result(object())

    assert isinstance(result, SerdeError)
    assert result.operation == "serialize_result"
    assert result.category == "unsupported_type"
    assert result.input_type == "object"
