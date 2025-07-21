from pathlib import Path
from typing import Optional

import click

from opencosmo.analysis.install import install_spec


@click.group()
def opencosmo():
    pass


@opencosmo.command(name="install")
@click.argument("spec_name", required=True)
@click.option("--file", type=click.Path(), required=False)
def install(spec_name: str, file: Optional[Path] = None):
    install_spec(spec_name)


if __name__ == "__main__":
    opencosmo()
