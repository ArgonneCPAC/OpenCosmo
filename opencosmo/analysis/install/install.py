import os
from typing import Optional

import networkx as nx

from .source import install_conda_forge, install_github, install_pip
from .specs import DependencySpec, get_specs

CONDA_ENV = os.environ.get("CONDA_DEFAULT_ENV")


def install_spec(name: str, versions: dict[str, Optional[str]] = {}):
    spec = get_specs()[name]
    requirements = spec.requirements
    raw_graph = {name: dep.depends_on for name, dep in requirements.items()}
    graph = nx.DiGraph(raw_graph).reverse()
    transaction = {}
    method: Optional[str] = None
    for requirement, data in requirements.items():
        if requirement not in versions:
            versions[requirement] = data.version

    for node in nx.topological_sort(graph):
        requirement_method = resolve_method(
            versions.get(node), requirements[node].prefer_source
        )
        if method is None or requirement_method == method:
            method = requirement_method
            transaction.update({node: versions.get(node)})
            continue
        execute_transaction(method, transaction, requirements)
        method = requirement_method
        transaction = {node: versions.get(node)}
    execute_transaction(method, transaction, requirements)


def resolve_method(version: Optional[str], method: Optional[str]):
    if version is None or "+g" not in version:
        if method is None and CONDA_ENV is not None:
            return "conda-forge"
        elif method != "conda-forge":
            return method
    elif "+g" in version:
        return "pip-git"
    return "pip"


def execute_transaction(
    method: Optional[str],
    transaction: dict[str, Optional[str]],
    requirements: dict[str, DependencySpec],
):
    if method == "conda-forge":
        install_conda_forge(transaction)
    elif method == "pip-git":
        for name, version in transaction.items():
            assert version is not None
            install_github(name, version, requirements)
    elif method == "pip":
        install_pip(transaction)
