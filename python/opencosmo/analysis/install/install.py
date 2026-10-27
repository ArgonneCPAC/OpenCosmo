from __future__ import annotations

import sys
from functools import cache
from graphlib import TopologicalSorter
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from .source import install_conda_forge, install_github, install_pip
from .specs import get_specs

if TYPE_CHECKING:
    from .specs import DependencySpec

logger = getLogger("opencosmo")


@cache
def is_conda_env():
    meta_path = Path(sys.prefix) / "conda-meta"
    return meta_path.exists()


@cache
def is_uv_env():
    cfg_path = Path(sys.prefix) / "pyvenv.cfg"

    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            # Look for a line starting with 'uv ='
            res = any(line.strip().startswith("uv =") for line in f)
            return res

    return False


def install_spec(
    name: str, versions: dict[str, str | None] | None = None, dev: bool = False
):
    if versions is None:
        versions = {}
    spec = get_specs()[name]
    logger.info(f"Installing analysis package {name}")
    requirements = spec.requirements

    dependency_graph = {
        req_name: set(dep.depends_on or ()) for req_name, dep in requirements.items()
    }
    if unknown := set().union(*dependency_graph.values()).difference(requirements):
        raise ValueError(
            f"Analysis package {name} depends on unknown packages: {sorted(unknown)}"
        )
    # Materialized up front so a CycleError is raised before anything is installed.
    install_order = list(TopologicalSorter(dependency_graph).static_order())

    transaction = {}
    method: str | None = None
    dev_transaction: dict[str, str | None] = {}

    for requirement, data in requirements.items():
        if requirement not in versions and not data.optional:
            versions[requirement] = data.version

    for nodename in install_order:
        if nodename not in versions:
            continue
        version = versions.get(nodename)
        if version is not None and "dev" in version:
            dev_transaction[nodename] = version
            continue

        requirement_method = resolve_method(
            versions.get(nodename), requirements[nodename].prefer_source
        )
        if method is None or requirement_method == method:
            method = requirement_method
            transaction.update({nodename: versions.get(nodename)})
            continue
        execute_transaction(method, transaction, requirements, dev)
        method = requirement_method
        transaction = {nodename: versions.get(nodename)}
    execute_transaction(method, transaction, requirements, dev)
    execute_transaction("pip-git", dev_transaction, requirements, dev)


def resolve_method(version: str | None, method: str | None):
    if version is None or "+g" not in version:
        if method in (None, "conda-forge") and is_conda_env():
            return "conda-forge"

    elif "+g" in version:
        return "pip-git"
    elif method is not None:
        return method
    return "pip"


def execute_transaction(
    method: str | None,
    transaction: dict[str, str | None],
    requirements: dict[str, DependencySpec],
    dev: bool = False,
):
    print(f"Installing {','.join(transaction.keys())}")
    if method == "conda-forge":
        install_conda_forge(transaction)
    elif method == "pip-git":
        for name, version in transaction.items():
            assert version is not None
            install_github(name, version, requirements)
    elif method == "pip":
        install_pip(transaction, dev)
