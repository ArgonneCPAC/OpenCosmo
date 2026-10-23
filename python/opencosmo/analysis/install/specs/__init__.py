import json
from functools import cache
from pathlib import Path

from pydantic import BaseModel


class DependencySpec(BaseModel):
    prefer_source: str
    repo: str | None = None
    depends_on: list[str] | None = []
    version: str | None = None
    optional: bool = False


class AnalysisSpec(BaseModel):
    name: str
    header_version_key: str | None = None
    requirements: dict[str, DependencySpec]


@cache
def get_specs() -> dict:
    dir = Path(__file__).parent
    files = dir.glob("*.json")
    specs = {}

    for file in files:
        try:
            name, spec = __load_spec(file)
        except (json.JSONDecodeError, ValueError):
            continue
        requirements = spec.pop("requirements")
        dep_specs = {
            name: DependencySpec(**spec_) for name, spec_ in requirements.items()
        }

        specs[name] = AnalysisSpec(**spec, requirements=dep_specs)

    return specs


def __load_spec(path: Path) -> tuple[str, dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if "name" not in data:
        raise ValueError

    return data["name"], data
