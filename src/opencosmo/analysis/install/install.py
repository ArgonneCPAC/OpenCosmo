from __future__ import annotations

import os
from logging import getLogger
from typing import TYPE_CHECKING, Optional

import networkx as nx

from .source import install_conda_forge, install_github, install_pip
from .specs import get_specs

if TYPE_CHECKING:
    from .specs import DependencySpec

CONDA_ENV = os.environ.get("CONDA_DEFAULT_ENV")
logger = getLogger("opencosmo")


def install_spec(name: str, versions: dict[str, Optional[str]] = {}, dev: bool = False):
    spec = get_specs()[name]
    logger.info(f"Installing analysis package {name}")
    requirements = spec.requirements
    raw_graph = {name: dep.depends_on for name, dep in requirements.items()}
    graph = nx.DiGraph(raw_graph).reverse()
    transaction = {}
    method: Optional[str] = None
    dev_transaction: dict[str, str | None] = {}

    for requirement, data in requirements.items():
        if requirement not in versions and not data.optional:
            versions[requirement] = data.version

    for node in nx.topological_sort(graph):
        if node not in versions:
            continue
        version = versions.get(node)
        if version is not None and "dev" in version:
            dev_transaction[node] = version
            continue

        requirement_method = resolve_method(
            versions.get(node), requirements[node].prefer_source
        )
        if method is None or requirement_method == method:
            method = requirement_method
            transaction.update({node: versions.get(node)})
            continue
        execute_transaction(method, transaction, requirements, dev)
        method = requirement_method
        transaction = {node: versions.get(node)}
    execute_transaction(method, transaction, requirements, dev)
    execute_transaction("pip-git", dev_transaction, requirements, dev)


def resolve_method(version: Optional[str], method: Optional[str]):
    if version is None or "+g" not in version:
        if method in (None, "conda-forge") and CONDA_ENV is not None:
            return "conda-forge"
    elif "+g" in version:
        return "pip-git"
    elif method is not None:
        return method
    return "pip"


def execute_transaction(
    method: Optional[str],
    transaction: dict[str, Optional[str]],
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
