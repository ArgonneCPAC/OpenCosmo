import subprocess
from typing import Optional


def install_conda_forge(packages: dict[str, Optional[str]]):
    command = "conda install -c conda-forge"
    for name, version in packages.items():
        command += f" {name}"
        if version is not None:
            command += f"={version}"

    run_install_command(command)


def install_github(repo_url: str, commit: Optional[str]):
    command = f"pip install {repo_url}"
    if commit is not None:
        command += f"@{commit}"

    run_install_command(command)


def run_install_command(command: str):
    cmd = command.split(" ")
    subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        check=True,
    )
