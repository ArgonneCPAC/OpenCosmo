from functools import cached_property
from typing import TYPE_CHECKING, Mapping, Optional

import astropy.units as u
import h5py
import numpy as np

from opencosmo.units import UnitConvention
from opencosmo.units.get import UnitApplicator, get_unit_applicators_hdf5

if TYPE_CHECKING:
    from opencosmo.header import OpenCosmoHeader


def make_unit_handler(
    group: h5py.Group,
    header: "OpenCosmoHeader",
    target_convention: Optional[UnitConvention] = None,
):
    applicators = get_unit_applicators_hdf5(group, header)
    if target_convention is None:
        target_convention = header.file.unit_convention
    return UnitHandler(header.file.unit_convention, target_convention, applicators)


class UnitHandler:
    def __init__(
        self,
        base_convention: UnitConvention,
        current_convention: UnitConvention,
        applicators: dict[str, UnitApplicator],
        conversions: dict[str, u.Unit] = {},
    ):
        self.__base_convention = base_convention
        self.__current_convention = current_convention
        self.__applicators = applicators
        self.__conversions = conversions

    @property
    def current_convention(self):
        return self.__current_convention

    @property
    def base_units(self):
        return {key: app.base_unit for key, app in self.__applicators.items()}

    @cached_property
    def unitless_columns(self):
        return set(
            key for key, app in self.__applicators.items() if app.base_unit is None
        )

    def with_static_columns(self, **columns: u.Unit):
        new_applicators = {}
        for colname, unit in columns.items():
            new_applicators[colname] = UnitApplicator.static(
                unit, self.__base_convention
            )
        return UnitHandler(
            self.__base_convention,
            self.__current_convention,
            self.__applicators | new_applicators,
            self.__conversions,
        )

    def verify_conversions(
        self, conversions: dict[str, u.Unit], convention: UnitConvention
    ):
        conversion_keys = set(conversions.keys())
        unitless_columns = self.unitless_columns.intersection(conversion_keys)
        if unitless_columns:
            raise ValueError(f"Columns {unitless_columns} do not carry units!")

        for name, unit in conversions.items():
            if not self.__applicators[name].can_convert(unit, convention):
                raise ValueError(
                    f"Cannot convert values with units {self.__applicators[name].unit_in_convention(convention)} to value with units {unit}"
                )

    def with_convention(self, convention: str | UnitConvention):
        convention = UnitConvention(convention)
        if convention == self.current_convention:
            return self
        return UnitHandler(self.__base_convention, convention, self.__applicators)

    def with_conversions(self, conversions: dict[str, u.Unit]):
        if not conversions:
            return self

        self.verify_conversions(conversions, self.__current_convention)
        new_conversions = self.__conversions | conversions
        return UnitHandler(
            self.__base_convention,
            self.__current_convention,
            self.__applicators,
            new_conversions,
        )

    def into_base_convention(
        self,
        data: Mapping[str, float | np.ndarray | u.Quantity],
        unit_kwargs: dict = {},
    ):
        return {
            name: self.__applicators[name].convert_to_base(
                val, self.__current_convention, unit_kwargs=unit_kwargs
            )
            for name, val in data.items()
        }

    def apply_units(self, data: dict[str, np.ndarray], unit_kwargs):
        columns = {}
        for key, value in data.items():
            applicator = self.__applicators.get(key)
            conversion = self.__conversions.get(key)
            if applicator is not None:
                columns[key] = applicator.apply(
                    value, self.__current_convention, unit_kwargs=unit_kwargs
                )
                if conversion is not None:
                    columns[key] = columns[key].to(conversion)
            else:
                columns[key] = value

        return columns
