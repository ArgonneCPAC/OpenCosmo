import astropy.cosmology.units as cu  # type: ignore
import astropy.units as u  # type: ignore

from .convention import UnitConvention

_ = u.add_enabled_units(cu)


__all__ = ["UnitConvention"]
