from __future__ import annotations

from typing import TYPE_CHECKING

import click

from opencosmo.analysis.install import get_file_versions, install_spec
from opencosmo.analysis.skill import (
    AGENT_TARGETS,
    SkillInstalled,
    SkillInstallFailed,
    install_skill,
)

if TYPE_CHECKING:
    from pathlib import Path


@click.command(name="install")
@click.argument("spec_name", required=True)
@click.option("--file", type=click.Path(exists=True), required=False)
@click.option("--dev", is_flag=True)
def install(spec_name: str, file: Path | None = None, dev: bool = False):
    if file is not None:
        versions = get_file_versions(spec_name, file)
    else:
        versions = {}

    install_spec(spec_name, versions, dev=dev)


@click.command(name="install-skill")
@click.argument("agent", type=click.Choice(sorted(AGENT_TARGETS)))
@click.option(
    "--local",
    is_flag=True,
    help="Install into the current directory's skills folder instead of globally.",
)
def install_skill_cli(agent: str, local: bool = False):
    """Install the OpenCosmo skill for a coding agent (pi, claude, opencode, codex)."""
    match install_skill(agent, local=local):
        case SkillInstalled(path=path, tag=tag):
            click.echo(f"Installed OpenCosmo skill (tag {tag}) to {path}")
        case SkillInstallFailed(reason=reason):
            raise click.ClickException(reason)
