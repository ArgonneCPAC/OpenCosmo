from pathlib import Path
from typing import Optional

import click

from opencosmo.analysis.install import get_file_versions, install_spec


@click.group()
def cli():
    pass


@cli.command(name="install")
@click.argument("spec_name", required=True)
@click.option("--file", type=click.Path(exists=True), required=False)
def install(spec_name: str, file: Optional[Path] = None):
    if file is not None:
        versions = get_file_versions(spec_name, file)
    else:
        versions = {}

    install_spec(spec_name, versions)


if __name__ == "__main__":
    cli()
