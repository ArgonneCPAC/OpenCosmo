import astropy.cosmology.units as cu
import astropy.units as u

from .convention import UnitConvention

_ = u.add_enabled_units(cu)


class UnitsError(Exception):
    pass


__all__ = ["UnitConvention", "UnitsError"]
