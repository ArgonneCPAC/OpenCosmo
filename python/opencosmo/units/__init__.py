import astropy.cosmology.units as cu  # type: ignore
import astropy.units as u  # type: ignore

from .convention import UnitConvention

_ = u.add_enabled_units(cu)


class UnitsError(Exception):
    pass


__all__ = ["UnitConvention", "UnitsError"]
