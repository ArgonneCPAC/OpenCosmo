import click

from opencosmo.analysis.cli import install
from opencosmo.io.cache import cache_layouts_cli


@click.group()
def cli():
    pass


cli.add_command(cache_layouts_cli)
cli.add_command(install)

if __name__ == "__main__":
    cli()
