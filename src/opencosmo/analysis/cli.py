from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import click

from opencosmo.analysis.install import get_file_versions, install_spec

if TYPE_CHECKING:
    from pathlib import Path


@click.group()
def cli():
    pass


@cli.command(name="install")
@click.argument("spec_name", required=True)
@click.option("--file", type=click.Path(exists=True), required=False)
@click.option("--dev", is_flag=True)
def install(spec_name: str, file: Optional[Path] = None, dev: bool = False):
    if file is not None:
        versions = get_file_versions(spec_name, file)
    else:
        versions = {}

    install_spec(spec_name, versions, dev=dev)


if __name__ == "__main__":
    cli()
