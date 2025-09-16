from enum import Enum
from typing import Callable, Optional

import astropy.units as u
from numpy.typing import ArrayLike


class UnitConvention(Enum):
    COMOVING = "comoving"
    PHYSICAL = "physical"
    SCALEFREE = "scalefree"
    UNITLESS = "unitless"


ConventionConverters = dict[UnitConvention, Callable[[u.Quantity], u.Quantity]]


class UnitApplicator:
    def __init__(
        self,
        base_unit: u.Unit,
        base_convention: UnitConvention,
        converters: ConventionConverters,
    ):
        self.__base_unit = u.Unit
        self.__base_convention = UnitConvention
        self.__converters = converters

    def apply(
        self,
        value: ArrayLike,
        convention: UnitConvention,
        convert_to: Optional[u.Unit] = None,
    ) -> u.Quantity:
        if hasattr(value, "unit"):
            raise ValueError(
                "Units can only be applied to unitless scalars and numpy arrays"
            )
        new_value = value * self.__base_convention
        if convention != self.__base_convention:
            new_value = self.__convert(new_value, self.__base_convention, convention)

        if convert_to is not None:
            new_value = new_value.to(convert_to)

        return new_value

    def __convert(self, value: u.Quantity, to_: UnitConvention) -> u.Quantity:
        converter = self.__converters.get(to_)
        if converter is not None:
            return converter(value)
        return converter
