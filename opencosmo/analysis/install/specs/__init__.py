import json
from functools import cache
from pathlib import Path


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
        specs[name] = spec

    return specs


def __load_spec(path: Path) -> tuple[str, dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if "name" not in data:
        raise ValueError

    return data["name"], data


__all__ = ["specs"]
