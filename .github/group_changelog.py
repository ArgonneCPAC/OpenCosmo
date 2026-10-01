"""Group final Towncrier releases by minor version."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


CHANGELOG_PATH = Path("docs/source/changelog.rst")
MARKER = ".. towncrier release notes start"
RELEASE_PATTERN = re.compile(
    r"^(?P<name>.+?) (?P<version>\d+\.\d+\.\d+) \((?P<date>[^\n]+)\)$"
)
UNDERLINE_PATTERN = re.compile(r"^(?P<character>[-=~^\"])(?P=character)*$")
MINOR_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")


def __is_underline(line: str) -> bool:
    return UNDERLINE_PATTERN.fullmatch(line) is not None


def __release_starts(lines: list[str]) -> list[tuple[int, tuple[int, int, int], str]]:
    starts: list[tuple[int, tuple[int, int, int], str]] = []
    for index, line in enumerate(lines[:-1]):
        match = RELEASE_PATTERN.fullmatch(line)
        if match is None or set(lines[index + 1]) not in ({"~"}, {"="}):
            continue
        version = tuple(int(part) for part in match["version"].split("."))
        starts.append((index, version, line))
    return starts


def __remove_minor_groups(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    index = 0
    while index < len(lines):
        if (
            MINOR_VERSION_PATTERN.fullmatch(lines[index]) is not None
            and index + 1 < len(lines)
            and set(lines[index + 1]) == {"-"}
        ):
            index += 2
            if index < len(lines) and not lines[index]:
                index += 1
            continue
        filtered.append(lines[index])
        index += 1
    return filtered


def __normalize_release(block: list[str], title: str) -> list[str]:
    normalized = [title, "~" * len(title), *block[2:]]
    for index, line in enumerate(normalized[:-1]):
        if UNDERLINE_PATTERN.fullmatch(normalized[index + 1]) is not None and set(
            normalized[index + 1]
        ) == {"-"}:
            normalized[index + 1] = "^" * len(normalized[index + 1])
    return normalized


def group_changelog(path: Path) -> None:
    """Group final release blocks in ``path`` by minor version."""
    lines = path.read_text().splitlines()
    try:
        marker_index = lines.index(MARKER)
    except ValueError as error:
        message = f"Towncrier marker {MARKER!r} was not found in {path}."
        raise ValueError(message) from error

    if marker_index < 3 or not __is_underline(lines[marker_index - 2]):
        message = f"A title must precede the Towncrier marker in {path}."
        raise ValueError(message)

    release_lines = __remove_minor_groups(lines[marker_index + 1 :])
    starts = __release_starts(release_lines)
    if not starts:
        message = f"No final X.Y.Z releases were found in {path}."
        raise ValueError(message)

    releases: dict[tuple[int, int], list[tuple[tuple[int, int, int], list[str]]]] = (
        defaultdict(list)
    )
    for position, (start, version, title) in enumerate(starts):
        end = (
            starts[position + 1][0]
            if position + 1 < len(starts)
            else len(release_lines)
        )
        block = __normalize_release(release_lines[start:end], title)
        while block and not block[-1]:
            block.pop()
        releases[version[:2]].append((version, block))

    groups: list[str] = []
    for minor_version in sorted(releases, reverse=True):
        minor_title = ".".join(str(part) for part in minor_version)
        groups.extend((minor_title, "-" * len(minor_title), ""))
        for _, block in sorted(releases[minor_version], reverse=True):
            groups.extend(block)
            groups.append("")

    output = [*lines[: marker_index + 1], "", *groups]
    path.write_text("\n".join(output).rstrip() + "\n")


if __name__ == "__main__":
    group_changelog(CHANGELOG_PATH)
