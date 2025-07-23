import shutil
import subprocess
import sys
from typing import Optional

from .specs import DependencySpec


def install_conda_forge(packages: dict[str, Optional[str]]):
    conda_executable = shutil.which("conda")

    prefix = sys.prefix
    if conda_executable is None:
        raise FileNotFoundError("Unable to locate the conda executable")

    command = f"{conda_executable} install --prefix {prefix} -y -c conda-forge"
    for name, version in packages.items():
        command += f" {name}"
        if version is not None:
            command += f"={version}"

    run_install_command(command)


def install_pip(packages: dict[str, Optional[str]]):
    command = f"{sys.executable} -m pip install"
    for name, version in packages.items():
        command += f" {name}"
        if version is not None:
            command += f"=={version}"
    run_install_command(command)


def install_github(name: str, version: str, dep_spec: dict[str, DependencySpec]):
    repo_url = dep_spec[name].repo
    assert repo_url is not None
    command = f"{sys.executable} -m pip install git+{repo_url}"
    commit = version.split("+g")[-1]
    command += f"@{commit}"

    run_install_command(command)


def run_install_command(command: str):
    print(command)
    _ = subprocess.run(command.split(" "))
