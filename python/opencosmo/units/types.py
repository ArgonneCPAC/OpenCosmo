from __future__ import annotations

import astropy.units as u

type AnyUnit = u.UnitBase | u.FunctionUnitBase
"""Any Astropy unit, including function units such as ``dex``."""

type UnitValue = AnyUnit | u.Quantity | None
"""
Value describing the unit of a column: a unit, a function unit (e.g. dex), a
Quantity (a unit with a prefactor, e.g. a constant), or None for unitless.
"""
