"""The message generator: reproducibility and rejection of unrepresentable APIs."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np  # noqa: TC002 - resolved by get_type_hints
import pytest
from annotated_types import Gt
from opencosmo.column.column import ConstructedColumn  # noqa: TC002
from opencosmo.serde.codegen.__main__ import main
from opencosmo.serde.codegen.config import SPECS, Check, MessageSpec, symbol
from opencosmo.serde.codegen.emit import render
from opencosmo.serde.live import to_rows


class Target:
    def unannotated(self, value) -> Target:
        return self

    def live_only(self, values: np.ndarray) -> Target:
        return self

    def unsupported_metadata(self, n: Annotated[int, Gt(0)]) -> Target:
        return self

    def mutable_default(self, names: dict[str, str] = {}) -> Target:  # noqa: B006
        return self

    def ambiguous_union(self, value: ConstructedColumn | np.ndarray | int) -> Target:
        return self

    def unresolved(self, value: Missing) -> Target:  # type: ignore[name-defined]  # noqa: F821
        return self

    def supported(
        self,
        n: int,
        *names: str,
        at: Literal["start", "end"] = "start",
        **columns: ConstructedColumn,
    ) -> Target:
        return self


def spec(tmp_path, *methods, checks=None):
    return MessageSpec(
        target=Target,
        public_name="Target",
        methods=methods,
        output=tmp_path / "target.py",
        namespace={},
        checks=checks or {},
    )


@pytest.mark.parametrize("message_spec", SPECS, ids=lambda s: s.target.__name__)
def test_checked_in_modules_are_current(message_spec):
    assert render(message_spec) == message_spec.output.read_text()


def test_check_mode_reports_current_modules():
    assert main(["--check"]) == 0


def test_generation_is_reproducible(tmp_path):
    first = render(spec(tmp_path, "supported"))

    assert first == render(spec(tmp_path, "supported"))
    assert "class TargetSupportedMessage(ExpressionModel):" in first
    assert "return target.supported(" in first
    assert 'reserved_keywords(self.columns, ("n", "at"))' in first


@pytest.mark.parametrize(
    ("method", "error"),
    [
        ("unannotated", r"unannotated\(value\) has no annotation"),
        ("live_only", r"live_only\(values\): .* can only be passed as a live value"),
        ("unsupported_metadata", "Unsupported annotation metadata"),
        ("mutable_default", "Unsupported default"),
        ("ambiguous_union", "convert differently"),
        ("unresolved", "add it to namespace"),
    ],
)
def test_generation_rejects_unrepresentable_signatures(tmp_path, method, error):
    with pytest.raises(TypeError, match=error):
        render(spec(tmp_path, method))


def test_checks_must_target_exposed_methods_and_fields(tmp_path):
    check = Check(symbol(to_rows), ("missing",))

    with pytest.raises(ValueError, match="unexposed methods"):
        render(spec(tmp_path, "supported", checks={"live_only": (check,)}))
    with pytest.raises(ValueError, match="unknown fields"):
        render(spec(tmp_path, "supported", checks={"supported": (check,)}))


def test_symbol_requires_an_importable_name():
    with pytest.raises(TypeError, match="no importable name"):
        symbol(3)

    class Local:
        pass

    with pytest.raises(TypeError, match="cannot be imported"):
        symbol(Local)
