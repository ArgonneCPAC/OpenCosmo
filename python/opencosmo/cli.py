import click

from opencosmo.analysis.cli import install, install_skill_cli
from opencosmo.io.cache import cache_layouts_cli


@click.group()
def cli():
    pass


cli.add_command(cache_layouts_cli)
cli.add_command(install)
cli.add_command(install_skill_cli)

if __name__ == "__main__":
    cli()
