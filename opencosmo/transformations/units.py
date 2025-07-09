from enum import Enum
from functools import partial
from typing import Optional

import astropy.cosmology.units as cu  # type: ignore
import astropy.units as u  # type: ignore
import h5py
from astropy.constants import m_p  # type: ignore
from astropy.cosmology import Cosmology
from astropy.table import Column, Table  # type: ignore

from opencosmo import transformations as t
from opencosmo.dataset.column import get_table_builder
from opencosmo.header import OpenCosmoHeader

_ = u.add_enabled_units(cu)

KNOWN_REDSHIFT_COLUMNS = {
    "redshift",
    "redshift_true",
}

KNOWN_SCALE_COLUMS = {"fof_halo_center_a"}


def get_a(table: Table):
    columns = set(table.columns)
    redshift_columns = columns.intersection(KNOWN_REDSHIFT_COLUMNS)
    if len(redshift_columns) == 1:
        redshifts = table[redshift_columns.pop()]
        return 1 / (redshifts + 1)

    scale_columns = columns.intersection(KNOWN_SCALE_COLUMS)

    if len(scale_columns) == 1:
        return table[scale_columns.pop()]
    raise ValueError("Unable to find redshift column to convert coordinates")


UNIT_MAP = {
    "comoving Mpc/h": u.Mpc / cu.littleh,
    "comoving (Mpc/h)^2": (u.Mpc / cu.littleh) ** 2,
    "comoving km/s": u.km / u.s,
    "comoving (km/s)^2": (u.km / u.s) ** 2,
    "Msun/h": u.Msun / cu.littleh,
    "Msun/yr": u.Msun / u.yr,
    "K": u.K,
    "comoving (Msun/h * (km/s) * Mpc/h)": (u.Msun / cu.littleh)
    * (u.km / u.s)
    * (u.Mpc / cu.littleh),
    "log10(erg/s)": u.DexUnit("erg/s"),
    "h^2 keV / (comoving cm)^3": (cu.littleh**2) * u.keV / (u.cm**3),
    "keV * cm^2": u.keV * u.cm**2,
    "cm^-3": u.cm**-3,
    "Gyr": u.Gyr,
    "Msun/h / (comoving Mpc/h)^3": (u.Msun / cu.littleh) / (u.Mpc / cu.littleh) ** 3,
    "Msun/h * km/s": (u.Msun / cu.littleh) * (u.km / u.s),
    "H0^-1": (u.s * (1 * u.Mpc).to(u.km).value).to(u.year) / (100 * cu.littleh),
    "m_hydrogen": m_p,
    "Msun * (km/s)^2": (u.Msun) * (u.km / u.s) ** 2,
}


class UnitError(Exception):
    pass


class UnitConvention(Enum):
    COMOVING = "comoving"
    PHYSICAL = "physical"
    SCALEFREE = "scalefree"
    UNITLESS = "unitless"


def get_unit_transformation_generators() -> list[t.TransformationGenerator]:
    """
    Get the unit transformation generato.

    We use generators for units because it is most appropriate to think
    of units as fundamental to the data, even when they don't actually
    appear in the hdf5 file.

    Even if the user requests unitless data, we still need to have
    access to these so that we can apply units if they
    call Dataset.with_convention later.
    """
    return [
        generate_attribute_unit_transformations,
    ]


def get_unit_transition_transformations(
    from_convention: str | UnitConvention,
    to_convention: str | UnitConvention,
    unit_applicators: t.TransformationDict,
    cosmology: Cosmology,
    redshift: float,
) -> t.TransformationDict:
    """
    Given a dataset, the user can request a transformation to a different unit
    convention. The returns a new set of transformations that will take the
    dataset to the requested unit convention.
    """
    from_ = UnitConvention(from_convention)
    to_ = UnitConvention(to_convention)
    if to_ == UnitConvention.UNITLESS:
        return {}
    match from_:
        case UnitConvention.SCALEFREE:
            update_transformations = get_scalefree_transition_transformations(
                to_, unit_applicators, cosmology, redshift
            )
        case UnitConvention.COMOVING:
            update_transformations = get_comoving_transition_transformations(
                to_, unit_applicators, cosmology, redshift
            )
        case UnitConvention.PHYSICAL:
            update_transformations = get_physical_transition_transformations(
                to_, unit_applicators, cosmology, redshift
            )
        case UnitConvention.UNITLESS:
            raise UnitError("Cannot convert units for datasets that have none!")

    for ttype in unit_applicators:
        existing = update_transformations.get(ttype, [])
        update_transformations[ttype] = unit_applicators[ttype] + existing
    return update_transformations


def get_comoving_transition_transformations(
    to: UnitConvention,
    unit_applicators: t.TransformationDict,
    cosmology: Cosmology,
    redshift: float,
):
    comoving_to_phys: t.TableTransformation = partial(
        comoving_to_physical, cosmology=cosmology, redshift=redshift
    )
    match to:
        case UnitConvention.PHYSICAL:
            return {t.TransformationType.TABLE: [comoving_to_phys]}
        case UnitConvention.COMOVING:
            return {}
        case UnitConvention.SCALEFREE:
            raise UnitError(
                "Data stored in comoving units cannot be converted into scalefree units"
            )


def get_physical_transition_transformations(
    to: UnitConvention,
    unit_applicators: t.TransformationDict,
    cosmology: Cosmology,
    redshift: float,
):
    phys_to_comoving: t.TableTransformation = partial(
        physical_to_comoving, cosmology=cosmology, redshift=redshift
    )
    match to:
        case UnitConvention.PHYSICAL:
            return {}
        case UnitConvention.COMOVING:
            return {t.TransformationType.TABLE: [phys_to_comoving]}
        case UnitConvention.SCALEFREE:
            raise UnitError(
                "Data stored in physical units cannot be converted into scalefree units"
            )


def get_scalefree_transition_transformations(
    to: UnitConvention,
    unit_applicators: t.TransformationDict,
    cosmology: Cosmology,
    redshift: float,
):
    remove_h: t.TableTransformation = partial(remove_littleh, cosmology=cosmology)
    comoving_to_phys: t.TableTransformation = partial(
        comoving_to_physical, cosmology=cosmology, redshift=redshift
    )
    match to:
        case UnitConvention.COMOVING:
            return {t.TransformationType.ALL_COLUMNS: [remove_h]}
        case UnitConvention.PHYSICAL:
            return {
                t.TransformationType.ALL_COLUMNS: [remove_h],
                t.TransformationType.TABLE: [comoving_to_phys],
            }
        case UnitConvention.SCALEFREE:
            return {}


def get_default_unit_transformations(
    file: h5py.File | h5py.Group, header: OpenCosmoHeader
):
    data_convention = UnitConvention(header.file.unit_convention)
    unit_applicators = get_base_unit_transformations(file["data"], header)
    match data_convention:
        case UnitConvention.SCALEFREE:
            base_transformations = get_unit_transition_transformations(
                data_convention,
                "comoving",
                unit_applicators,
                header.cosmology,
                header.file.redshift,
            )
        case UnitConvention.UNITLESS:
            base_transformations = {}
        case UnitConvention.COMOVING:
            base_transformations = unit_applicators
        case UnitConvention.PHYSICAL:
            raise NotImplementedError()

    column_names = list(str(col) for col in file["data"].keys())
    builder = get_table_builder(base_transformations, column_names)

    return builder, unit_applicators


def get_base_unit_transformations(
    input: h5py.Dataset,
    header: OpenCosmoHeader,
) -> t.TransformationDict:
    """
    Get the base unit transformations for a given dataset. These transformations
    produce the units that the data are actually stored in. Datasets alwyas
    hold onto a copy of these transformations even if the user later requests
    a different unit convention.


    These always apply after the initial transformations generated above.
    """
    generators = get_unit_transformation_generators()
    base_transformations = t.generate_transformations(input, generators, {})
    return base_transformations


def remove_littleh(column: Column, cosmology: Cosmology) -> Optional[Table]:
    """
    Remove little h from the units of the input table. For comoving
    coordinates, this is the second step after parsing the units themselves.
    """
    if (unit := column.unit) is not None:
        # Handle dex units
        try:
            if isinstance(unit, u.DexUnit):
                u_base = unit.physical_unit
                constructor = u.DexUnit
            else:
                u_base = unit

                def constructor(x):
                    return x
        except AttributeError:
            return None

        try:
            index = u_base.bases.index(cu.littleh)
        except ValueError:
            return None
        power = u_base.powers[index]
        new_unit = constructor(u_base / cu.littleh**power)
        column = column.to(new_unit, cu.with_H0(cosmology.H0))
    return column


def comoving_to_physical(
    table: Table, cosmology: Cosmology, redshift: float
) -> Optional[Table]:
    """
    Convert comoving coordinates to physical coordinates. This is the
    second step after parsing the units themselves.
    """
    try:
        a = get_a(table)
    except ValueError:
        a = cosmology.scale_factor(redshift)

    for colname in table.columns:
        if (unit := table[colname].unit) is not None:
            # Check if the units have distances in them
            decomposed = unit.decompose()
            try:
                index = decomposed.bases.index(u.m)
            except (ValueError, AttributeError):
                continue
            power = decomposed.powers[index]
            # multiply by the scale factor to the same power as the distance
            table[colname] = table[colname] * a**power

    return table


def physical_to_comoving(
    table: Table, cosmology: Cosmology, redshift: float
) -> Optional[Table]:
    """
    Convert comoving coordinates to physical coordinates. This is the
    second step after parsing the units themselves.
    """

    try:
        a = get_a(table)
    except ValueError:
        a = cosmology.scale_factor(redshift)

    for colname in table.columns:
        if (unit := table[colname].unit) is not None:
            # Check if the units have distances in them
            decomposed = unit.decompose()
            try:
                index = decomposed.bases.index(u.m)
            except (ValueError, AttributeError):
                continue
            power = decomposed.powers[index]
            # divide by the scale factor to the same power as the distance
            table[colname] = table[colname] / a**power

    return table


def get_raw_units(column: h5py.Dataset):
    if "unit" in column.attrs:
        if (us := column.attrs["unit"]) == "None" or us == "":
            return None
        if (unit := UNIT_MAP.get(us)) is not None:
            return unit
        try:
            return u.Unit(us)
        except ValueError:
            return None

    return None


def generate_attribute_unit_transformations(
    column: h5py.Dataset,
) -> t.TransformationDict:
    """
    Check the attributes of an hdf5 dataset to see if information about units is stored
    there.

    The raw HACC data does not store units in this way, relying instead on standard
    naming conventions. However if a user creates a new column, we want to be able to
    store it in our standard format without losing unit information and we cannot rely
    on them following our naming conventions.
    """
    unit = get_raw_units(column)
    if unit is not None:
        apply_func: t.Transformation = apply_unit(
            column_name=column.name.split("/")[-1], unit=unit
        )
        return {t.TransformationType.COLUMN: [apply_func]}
    return {}


class apply_unit:
    """
    Apply a unit to an input column. Ensuring that the correct column is
    passed will be the responsibility of the caller.

    Has to be a class so it can implement the ColumnTransformation protocol.
    """

    def __init__(self, column_name: str, unit: u.Unit):
        self.__name = column_name
        self.unit = unit

    def __call__(self, input: Column | float) -> Optional[Column | float]:
        if isinstance(input, float) or input.unit is None:
            return input * self.unit
        if input.unit is None:
            return input * self.unit
        return input

    @property
    def column_name(self) -> str:
        return self.__name
