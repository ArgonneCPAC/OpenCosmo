from collections.abc import Callable
from enum import Enum

import astropy.units as u


class UnitConvention(Enum):
    COMOVING = "comoving"
    PHYSICAL = "physical"
    SCALEFREE = "scalefree"
    UNITLESS = "unitless"


ConventionConverters = dict[UnitConvention, Callable[[u.Quantity], u.Quantity]]
