"""Download the OpenCosmo agent skill from its GitHub release and install it for a coding agent."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import opencosmo

REPO = "ArgonneCPAC/OpenCosmo"
ASSET_NAME = "opencosmo-skill.tar.gz"
SKILL_NAME = "opencosmo-library"
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class AgentTarget:
    """Where a given coding agent looks for skills."""

    global_dir: Path
    local_dir: Path


# The cross-agent ``.agents/skills`` convention, read by pi, OpenCode, and Codex.
__AGENTS_STANDARD = AgentTarget(
    global_dir=Path("~/.agents/skills"),
    local_dir=Path(".agents/skills"),
)

# Claude Code does not read ``.agents/skills`` and needs its own location.
__CLAUDE = AgentTarget(
    global_dir=Path("~/.claude/skills"),
    local_dir=Path(".claude/skills"),
)

AGENT_TARGETS: dict[str, AgentTarget] = {
    "pi": __AGENTS_STANDARD,
    "opencode": __AGENTS_STANDARD,
    "codex": __AGENTS_STANDARD,
    "claude": __CLAUDE,
}


@dataclass(frozen=True)
class SkillInstalled:
    path: Path
    tag: str


@dataclass(frozen=True)
class SkillInstallFailed:
    reason: str


def release_tag(version: str) -> str:
    """
    Git tag for a library version.

    Files carry pre-releases as ``1.5.0-b1`` (see ``.bumpversion.toml``) while the
    tag created by the release workflow is ``1.5.0b1``.
    """
    return version.replace("-", "")


def asset_url(version: str) -> str:
    """URL of the skill archive attached to the GitHub release for ``version``."""
    tag = release_tag(version)
    return f"https://github.com/{REPO}/releases/download/{tag}/{ASSET_NAME}"


def resolve_destination(agent: str, local: bool, cwd: Path | None = None) -> Path:
    target = AGENT_TARGETS[agent]
    if local:
        return (cwd or Path.cwd()) / target.local_dir / SKILL_NAME
    return target.global_dir.expanduser() / SKILL_NAME


def __extract_skill(response: IO[bytes], staging: Path) -> Path | None:
    """Stream the release asset into ``staging``. Returns the skill directory."""
    with tarfile.open(fileobj=response, mode="r|gz") as tar:
        tar.extractall(staging, filter="data")
    skill_dir = staging / SKILL_NAME
    return skill_dir if (skill_dir / "SKILL.md").is_file() else None


def __download_skill(version: str, staging: Path) -> Path | SkillInstallFailed:
    url = asset_url(version)
    request = urllib.request.Request(url, headers={"User-Agent": "opencosmo"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            skill_dir = __extract_skill(response, staging)
    except urllib.error.HTTPError as e:
        return SkillInstallFailed(f"{url} returned HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError, tarfile.TarError) as e:
        return SkillInstallFailed(f"Unable to download {url}: {e}")
    if skill_dir is None:
        return SkillInstallFailed(f"{url} does not contain a '{SKILL_NAME}/SKILL.md'")
    return skill_dir


def install_skill(
    agent: str,
    local: bool = False,
    version: str | None = None,
    cwd: Path | None = None,
) -> SkillInstalled | SkillInstallFailed:
    """
    Install the skill matching the current library version for ``agent``.

    Downloads the skill archive published as an asset of the GitHub release for
    the version's tag and replaces any existing installation.
    """
    if agent not in AGENT_TARGETS:
        supported = ", ".join(sorted(AGENT_TARGETS))
        return SkillInstallFailed(f"Unknown agent '{agent}'. Supported: {supported}")

    version = version or opencosmo.__version__
    destination = resolve_destination(agent, local, cwd)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".skill-") as tmp:
        result = __download_skill(version, Path(tmp))
        if isinstance(result, SkillInstallFailed):
            return SkillInstallFailed(
                f"Could not install the skill for opencosmo {version}: {result.reason}"
            )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(result, destination)
    return SkillInstalled(path=destination, tag=release_tag(version))
