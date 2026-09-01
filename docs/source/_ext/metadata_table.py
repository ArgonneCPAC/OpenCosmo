"""Sphinx extension that documents the dynamic header/dataset metadata blocks.

``OpenCosmoHeader`` builds its attribute table at runtime from the
``ACCESS_PATH``/``PARAMETER_ACCESS_PATHS`` declared on each parameter model
registered for a given file origin and data type (see
``opencosmo.dtypes.origin`` and ``opencosmo.dtypes.dtype``). Because that set
of models is data-dependent, it cannot be represented as static attributes on
``Dataset``/``OpenCosmoHeader`` for autodoc to pick up.

This extension introspects the same registries used by
``opencosmo.header.get_access_table`` and renders a table of every known
access path, which origins/data types expose it, and a link to the
corresponding Pydantic model documentation. Because it reads directly from
the registries, the generated table can never drift from the actual runtime
behavior of ``header.parameters``/``dataset.<name>``.

Usage in a .rst file::

    .. metadata-table::

"""

from __future__ import annotations

from dataclasses import dataclass, field

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


@dataclass
class _AccessPathInfo:
    origins: set[str] = field(default_factory=set)
    dtypes: set[str] = field(default_factory=set)
    required: bool = False
    models: dict[str, type] = field(default_factory=dict)


def _model_ref(model: type) -> str:
    return f":py:class:`~{model.__module__}.{model.__qualname__}`"


def _register(
    table: dict[str, _AccessPathInfo],
    model: type,
    *,
    origin: str,
    dtype: str | None,
    required: bool,
) -> None:
    paths = []
    if hasattr(model, "PARAMETER_ACCESS_PATHS"):
        paths.extend(model.PARAMETER_ACCESS_PATHS.values())
    if hasattr(model, "ACCESS_PATH"):
        paths.append(model.ACCESS_PATH)

    for path in paths:
        entry = table.setdefault(path, _AccessPathInfo())
        entry.origins.add(origin)
        if dtype is not None:
            entry.dtypes.add(dtype)
        entry.required = entry.required or required
        entry.models[f"{model.__module__}.{model.__qualname__}"] = model


def _iter_union_members(model_or_union) -> list[type]:
    args = getattr(model_or_union, "__args__", None)
    return list(args) if args is not None else [model_or_union]


def _build_access_path_table() -> dict[str, _AccessPathInfo]:
    """Walk the live parameter registries and collect every access path.

    Mirrors the traversal ``opencosmo.header.OpenCosmoHeader`` performs when
    it builds its access table, so this table is generated from the same
    source of truth rather than hand-maintained.
    """
    from opencosmo.dtypes.dtype import get_dtype_parameters
    from opencosmo.dtypes.file import DatasetType, FileParameters
    from opencosmo.dtypes.origin import get_origin_parameters

    table: dict[str, _AccessPathInfo] = {}

    # `file` block parameters (dtype, redshift, ...) apply to every file.
    _register(table, FileParameters, origin="all", dtype=None, required=True)

    known_origins = ["HACC"]
    for origin in known_origins:
        origin_params = get_origin_parameters(origin)
        for required, models in origin_params.items():
            is_required = required == "required"
            for model in models.values():
                for concrete in _iter_union_members(model):
                    _register(
                        table,
                        concrete,
                        origin=origin,
                        dtype=None,
                        required=is_required,
                    )

        for dtype_enum in DatasetType:
            for is_lightcone in (False, True):
                file_pars = FileParameters(
                    origin=origin,
                    data_type=dtype_enum.value,
                    is_lightcone=is_lightcone,
                )
                dtype_params = get_dtype_parameters(file_pars)
                for required, models in dtype_params.items():
                    is_required = required == "required"
                    for model in models.values():
                        for concrete in _iter_union_members(model):
                            _register(
                                table,
                                concrete,
                                origin=origin,
                                dtype=dtype_enum.value,
                                required=is_required,
                            )

    return table


class MetadataTableDirective(SphinxDirective):
    """Render a table of every metadata block known to the header system."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        table = _build_access_path_table()

        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 15 15 15 15 40",
            "",
            "   * - Attribute Name",
            "     - Dataset source",
            "     - Dataset type(s)",
            "     - Always present",
            "     - Model",
        ]
        for path in sorted(table):
            info = table[path]
            origins = ", ".join(sorted(info.origins)) or "any"
            dtypes = ", ".join(sorted(info.dtypes)) or "any"
            required = "Yes" if info.required else "No"
            model_refs = ", ".join(
                _model_ref(model) for _, model in sorted(info.models.items())
            )
            lines.extend(
                [
                    f"   * - ``{path}``",
                    f"     - {origins}",
                    f"     - {dtypes}",
                    f"     - {required}",
                    f"     - {model_refs}",
                ]
            )

        result = StringList(lines)
        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(result, 0, node)
        return node.children


def setup(app):
    app.add_directive("metadata-table", MetadataTableDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
