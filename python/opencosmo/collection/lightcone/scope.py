from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from opencosmo.column.column import RawColumn, _contains_scalar

if TYPE_CHECKING:
    from uuid import UUID

    from opencosmo.column.column import ConstructedColumn


@dataclass(frozen=True)
class SelectPlan:
    """
    The routing decision for one ``Lightcone.select(...)`` call. Returned by
    ``LightconeScope.plan_select``; consumed by ``Lightcone.select`` to drive
    the per-child ``select`` calls and construct the resulting scope.

    Attributes
    ----------
    new_scope : LightconeScope
        The scope to attach to the resulting lightcone.
    child_positional : set[str]
        Positional column names to forward to each child's ``select``.
    child_additional : set[str]
        Raw columns each child must read so the scope's dependencies can be
        resolved at ``get_data`` time (a superset of ``child_positional``
        when scope expressions read columns the user did not ask for).
    child_scoped : dict[str, ConstructedColumn]
        Derived kwargs that are *not* scope-owned and should be forwarded
        to each child's ``select`` as derived columns.
    hidden_additions : set[str]
        Columns added solely to support scope evaluation; the lightcone
        hides these from output.
    """

    new_scope: LightconeScope
    child_positional: set[str]
    child_additional: set[str]
    child_scoped: dict[str, ConstructedColumn] = field(default_factory=dict)
    hidden_additions: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LightconeScope:
    """
    Lightcone-level derived columns. Holds expressions whose evaluation must
    span all children (anything containing a DerivedScalarValue, or anything
    that depends on a name already in the scope).
    """

    producers: tuple[ConstructedColumn, ...] = ()
    column_map: dict[str, UUID] = None  # type: ignore
    descriptions: dict[str, str | None] = None  # type: ignore

    def __post_init__(self):
        if self.column_map is None:
            object.__setattr__(self, "column_map", {})
        if self.descriptions is None:
            object.__setattr__(self, "descriptions", {})

    def names(self) -> set[str]:
        """Return the names of all scope-owned columns."""
        return set(self.column_map.keys())

    @property
    def derived_producers(self) -> tuple[ConstructedColumn, ...]:
        """
        Producers that own at least one name in ``column_map`` — the actual
        scope-owned expressions a caller asked for. Excludes the synthetic
        ``RawColumn`` placeholders carried for graph validation.
        """
        return tuple(
            p
            for p in self.producers
            if any(name in self.column_map for name in (p.produces or ()))
        )

    @property
    def placeholders(self) -> tuple[RawColumn, ...]:
        """
        Synthetic ``RawColumn`` nodes added so ``build_dependency_graph`` can
        resolve scope expressions' UUID-keyed dependencies on raw child
        columns. Not user-visible.
        """
        return tuple(p for p in self.producers if isinstance(p, RawColumn))

    def add(
        self,
        new_columns: dict[str, ConstructedColumn],
        child_columns: set[str],
    ) -> LightconeScope:
        """
        Add new columns to the scope. Separates scope-owned (those containing scalars
        or depending on existing scope names) from child-scoped (pure child columns).
        Returns a new LightconeScope with the scoped columns added.

        The new producers are bound against a name→UUID map covering both the
        existing scope columns and synthetic ``RawColumn`` placeholders for the
        raw child columns each expression reads. Placeholders are stored in
        ``producers`` so the dependency graph validates cleanly when
        ``Lightcone.get_data`` runs ``evaluate_producers``.

        Parameters
        ----------
        new_columns : dict[str, ConstructedColumn]
            The new columns to consider.
        child_columns : set[str]
            The set of child column names (raw columns from children).

        Returns
        -------
        LightconeScope
            A new scope with scoped columns added.
        """
        scoped, _ = partition_columns(new_columns, child_columns, self)
        if not scoped:
            return self

        existing_placeholders: dict[str, RawColumn] = {
            p.name: p for p in self.placeholders
        }
        derived_producers = list(self.derived_producers)

        name_to_uuid: dict[str, UUID] = dict(self.column_map)
        for name, placeholder in existing_placeholders.items():
            name_to_uuid[name] = placeholder.uuid

        new_placeholders: dict[str, RawColumn] = {}
        for name, col in scoped.items():
            if not hasattr(col, "_traverse_names"):
                continue
            for raw_name in col._traverse_names():
                if raw_name in name_to_uuid:
                    continue
                placeholder = RawColumn(raw_name, None)
                new_placeholders[raw_name] = placeholder
                name_to_uuid[raw_name] = placeholder.uuid

        new_derived: list[ConstructedColumn] = []
        new_column_map = dict(self.column_map)
        new_descriptions = dict(self.descriptions)
        for name, col in scoped.items():
            col.name = name  # type: ignore[attr-defined]
            bound = col.bind(name_to_uuid)
            new_derived.append(bound)
            new_column_map[name] = bound.uuid
            new_descriptions[name] = bound.description

        new_producers = (
            list(existing_placeholders.values())
            + list(new_placeholders.values())
            + derived_producers
            + new_derived
        )
        return LightconeScope(tuple(new_producers), new_column_map, new_descriptions)

    def select(self, columns: set[str]) -> LightconeScope:
        """
        Restrict the scope to only the given column names.
        Returns a new LightconeScope with only the selected columns.

        Parameters
        ----------
        columns : set[str]
            The scope column names to keep.

        Returns
        -------
        LightconeScope
            A new scope with only selected columns (and their dependencies).
        """
        scope_names = self.names()
        selected = columns.intersection(scope_names)
        if not selected:
            return LightconeScope()

        keep_names = set(selected)

        for producer in self.producers:
            prod_names = producer.produces or set()
            for name in prod_names:
                if name in selected:
                    break
            else:
                continue

            for dep_uuid in producer.requires:
                for other in self.producers:
                    if other.uuid == dep_uuid:
                        other_names = other.produces or set()
                        if other_names:
                            keep_names.update(other_names)

        new_producers = [
            col
            for col in self.producers
            if any(name in keep_names for name in (col.produces or set()))
        ]
        new_column_map = {
            name: uuid for name, uuid in self.column_map.items() if name in keep_names
        }
        new_descriptions = {
            name: desc for name, desc in self.descriptions.items() if name in keep_names
        }

        return LightconeScope(tuple(new_producers), new_column_map, new_descriptions)

    def required_child_columns(self) -> set[str]:
        """
        Return the union of every raw/child column that the scoped expressions read.
        These are the columns that must be present in each child for scope evaluation.
        """
        return {p.name for p in self.placeholders}

    def plan_select(
        self,
        positional: set[str],
        derived_columns: dict[str, ConstructedColumn],
        raw_child_columns: set[str],
    ) -> SelectPlan:
        """
        Decide how to route a ``Lightcone.select(...)`` call: which derived
        kwargs are scope-owned vs child-scoped, what the resulting scope
        looks like, which positional columns hit the children directly, and
        which extra raw columns each child must read (and then hide) so the
        scope's dependencies can be resolved at evaluation time.

        Parameters
        ----------
        positional : set[str]
            Column names passed positionally to ``select``.
        derived_columns : dict[str, ConstructedColumn]
            Derived columns passed as kwargs to ``select``.
        raw_child_columns : set[str]
            Raw column names available on each child dataset.

        Returns
        -------
        SelectPlan
            The routing decision; see field docstrings on ``SelectPlan``.
        """
        scoped, child_scoped = partition_columns(
            derived_columns, raw_child_columns, self
        )

        previous_scope_names = self.names()
        kept_scope_names = (positional & previous_scope_names) | set(scoped.keys())
        new_scope = self.select(kept_scope_names).add(
            derived_columns, raw_child_columns
        )

        child_positional = positional - previous_scope_names - set(scoped.keys())
        scope_deps = new_scope.required_child_columns()
        hidden_additions = scope_deps - child_positional

        return SelectPlan(
            new_scope=new_scope,
            child_positional=child_positional,
            child_additional=set(scope_deps),
            child_scoped=child_scoped,
            hidden_additions=hidden_additions,
        )

    def scalar_names(self) -> set[str]:
        """Names of scope-owned columns that evaluate to a scalar."""
        from opencosmo.column.column import produces_scalar

        return {
            name
            for producer in self.derived_producers
            for name in (producer.produces or set())
            if produces_scalar(producer)
        }


def partition_columns(
    new_columns: dict[str, ConstructedColumn],
    child_columns: set[str],
    scope: LightconeScope | None = None,
) -> tuple[dict[str, ConstructedColumn], dict[str, ConstructedColumn]]:
    """
    Classify new columns into scoped (contain scalars or depend on scope names) and child-scoped.

    Parameters
    ----------
    new_columns : dict[str, ConstructedColumn]
        The columns to partition.
    child_columns : set[str]
        The child column names (raw columns from children).
    scope : LightconeScope, optional
        The current scope (if any). Columns depending on scope names are scoped.

    Returns
    -------
    tuple[dict, dict]
        (scoped_columns, child_scoped_columns)
    """
    if scope is None:
        scope = LightconeScope()

    scoped = {}
    child_scoped = {}
    scope_names = scope.names()

    for name, col in new_columns.items():
        if _contains_scalar(col):
            scoped[name] = col
        else:
            col_deps = set()
            if hasattr(col, "_traverse_names"):
                col_deps = col._traverse_names()
            if col_deps.intersection(scope_names):
                scoped[name] = col
            else:
                child_scoped[name] = col

    return scoped, child_scoped
