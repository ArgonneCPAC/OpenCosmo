import subprocess
from typing import Optional

from .specs import DependencySpec


def install_conda_forge(packages: dict[str, Optional[str]]):
    command = "conda install -c conda-forge"
    for name, version in packages.items():
        command += f" {name}"
        if version is not None:
            command += f"={version}"

    run_install_command(command)


def install_pip(packages: dict[str, Optional[str]]):
    command = "pip install"
    for name, version in packages.items():
        command += f" {name}"
        if version is not None:
            command += f"=={version}"
    run_install_command(command)


def install_github(name: str, version: str, dep_spec: dict[str, DependencySpec]):
    repo_url = dep_spec[name].repo
    assert repo_url is not None
    command = f"pip install git+{repo_url}"
    commit = version.split("+g")[-1]
    command += f"@{commit}"

    run_install_command(command)


def run_install_command(command: str):
    cmd = ["bash", "-l", "-c", command]
    _ = subprocess.run(cmd)
