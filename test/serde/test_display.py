"""Tests for the user-facing summary representations."""

import pytest
from opencosmo.serde import ResultSummary, format_region, format_summary
from pydantic import TypeAdapter

HEADER = {
    "origin": "HACC",
    "data_type": "halo_properties",
    "is_lightcone": False,
    "redshift": None,
    "step": None,
    "file_unit_convention": "comoving",
}


def __dataset(length=3, columns=(), region=None, sorted_by=None):
    return {
        "kind": "Dataset",
        "length": length,
        "columns": list(columns),
        "header": HEADER,
        "current_unit_convention": "physical",
        "sorted_by": sorted_by,
        "region": region,
    }


def __build(payload) -> ResultSummary:
    return TypeAdapter(ResultSummary).validate_python(payload)


def test_dataset_lists_columns_region_and_sorting():
    text = format_summary(
        __build(
            __dataset(
                columns=[
                    {"name": "mass", "description": "Halo mass", "unit": "Msun"},
                    {"name": "id", "description": None, "unit": None},
                ],
                region={"kind": "cone", "center": [10.0, -5.0], "radius": 2.0},
                sorted_by="mass",
            )
        )
    )
    lines = text.split("\n")
    assert lines[0] == "OpenCosmo Dataset (length=3)"
    assert "  mass [Msun]: Halo mass" in lines
    assert "  id" in lines
    assert "Unit convention: physical" in lines
    assert "Sorted by: mass" in lines
    assert any(line.startswith("Region: Cone Region") for line in lines)


def test_long_column_lists_are_truncated():
    columns = [{"name": f"c{i}", "description": None, "unit": None} for i in range(15)]
    text = format_summary(__build(__dataset(columns=columns)))
    assert "Columns (15):" in text
    assert "  ... and 5 more" in text
    assert "c10" not in text


def test_simulation_collection_counts_and_members():
    text = format_summary(
        __build(
            {
                "kind": "SimulationCollection",
                "length": 2,
                "members": [
                    {"key": "a", "value": __dataset()},
                    {"key": "b", "value": __dataset(length=7)},
                ],
            }
        )
    )
    assert text.split("\n")[0] == "SimulationCollection(0 collections, 2 datasets)"
    assert "  b: OpenCosmo Dataset (length=7)" in text


def test_structure_collection_header():
    text = format_summary(
        __build(
            {
                "kind": "StructureCollection",
                "length": 4,
                "data_type": "halo_properties",
                "header": HEADER,
                "region": None,
                "sorted_by": None,
                "members": [{"key": "halo_properties", "value": __dataset()}],
            }
        )
    )
    assert text.split("\n")[0] == "Collection of halos (length=4)"


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        (
            {"kind": "skybox", "p1": [0.0, -1.0], "p2": [2.0, 1.0]},
            "Sky Box Region (RA: 0.0000°–2.0000°, Dec: -1.0000°–1.0000°)",
        ),
        (
            {"kind": "healpix", "nside": 64, "pixel_count": 5},
            "Healpix Region (nside = 64, 5 pixels)",
        ),
        (
            {"kind": "box", "p1": [0.0, 0.0, 0.0], "p2": [1.0, 1.0, 1.0]},
            "Box with bounds [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]",
        ),
    ],
)
def test_format_region(region, expected):
    summary = __build(__dataset(region=region))
    assert summary.region is not None
    assert format_region(summary.region) == expected
