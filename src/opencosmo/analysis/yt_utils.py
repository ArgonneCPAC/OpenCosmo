from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import yt  # type: ignore
from pyxsim import CIESourceModel  # type: ignore
from unyt import unyt_array, unyt_quantity  # type: ignore

if TYPE_CHECKING:
    from yt.data_objects.static_output import Dataset as YT_Dataset

    import opencosmo as oc

# ---- define some constants ---- #
mp = unyt_quantity(1.67262192595e-24, "g")
kB = unyt_quantity(1.380649e-16, "erg/K")
solar_metallicity = 0.012899  # value used internally in HACC
# ------------------------------- #


def create_yt_dataset(
    data: Dict[str, oc.Dataset],
    compute_xray_fields: Optional[bool] = False,
    return_source_model: Optional[bool] = False,
    source_model_kwargs: Optional[Dict[str, Any]] = {},
) -> Union[YT_Dataset, Tuple[YT_Dataset, CIESourceModel]]:
    """
    Converts particle data to a `yt` dataset. Note that `yt`
    is generally developed with AMR codes in mind, but support for
    SPH codes is continually being added. `yt's` documentation can
    be found `here <https://yt-project.org/doc/index.html>`_.

    If `compute_xray_fields` is enabled, X-ray emissivity and luminosity fields
    will be added using `pyxsim <https://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/index.html#>`_,
    which generates photon samples from gas properties.

    Parameters
    ----------
    data : dict of astropy.table.Table
        A dictionary of particle datasets. Must include at least positions and masses.
    compute_xray_fields : bool, optional
        Whether or not to compute X-ray luminosities with `pyxsim`.
        Uses `CIESourceModel`, which considers thermal emission from gas assuming
        collisional ionization equilibrium.
    return_source_model : bool, optional
        Whether or not to return the `pyxsim` source model for further interaction,
        such as computing additional luminosities in different frequency bands
        or generating synthetic observations.
    source_model_kwargs : dict, optional
        Keyword arguments passed to the `CIESourceModel` constructor in `pyxsim`.
        These can include parameters like `emin`, `emax`, `nbins`, `abund_table`, etc.,
        to control the spectral resolution and emission model behavior. If `None`,
        default values will be used for all source model parameters.

    Returns
    -------
    ds : yt.data_objects.static_output.Dataset
        A `yt` dataset built from the input particle data, with additional fields
        (e.g., X-ray luminosities) if requested.

    source_model : pyxsim.source_models.CIESourceModel, optional
        Returned only if `return_source_model=True`.
    """

    data_dict: Dict[
        Union[Tuple[str, str], str], Union[np.ndarray, Tuple[np.ndarray, str]]
    ] = {}

    minx, maxx = np.inf, -np.inf
    miny, maxy = np.inf, -np.inf
    minz, maxz = np.inf, -np.inf

    # Fields that we need to rename to hook up to yt's internals
    special_fields = {
        "x": "particle_position_x",
        "y": "particle_position_y",
        "z": "particle_position_z",
        "mass": "particle_mass",
        "rho": "density",
        "hh": "smoothing_length",
    }

    def astropy_to_yt(array):
        """
        Converts from astropy format to yt format.
        Basically just reformats the units.
        """

        if array.unit is None:
            return unyt_array(array.data, "dimensionless")

        return unyt_array.from_astropy(array)

    for ptype in data.keys():
        if "particles" not in ptype:
            continue

        particle_data = data[ptype].data
        redshift = data[ptype].redshift
        cosmo = data[ptype].cosmology
        ptype_short = ptype.split("_")[0]

        for field in particle_data.keys():
            yt_field_name = special_fields.get(field, field)
            yt_particle_data = astropy_to_yt(particle_data[field])

            data_dict[(ptype_short, yt_field_name)] = (
                np.asarray(yt_particle_data.d),
                str(yt_particle_data.units),
            )

        minx, maxx = (
            min(minx, min(particle_data["x"].value)),
            max(maxx, max(particle_data["x"].value)),
        )
        miny, maxy = (
            min(miny, min(particle_data["y"].value)),
            max(maxy, max(particle_data["y"].value)),
        )
        minz, maxz = (
            min(minz, min(particle_data["z"].value)),
            max(maxz, max(particle_data["z"].value)),
        )

    bbox = [[minx, maxx], [miny, maxy], [minz, maxz]]

    # Raise error if any bound is still infinite
    if any(np.isinf(bound) for axis in bbox for bound in axis):
        raise ValueError(
            "Bounding box coordinates contain infinite values."
            "Check input data for missing or invalid positions."
        )

    ds = yt.load_particles(
        data_dict,
        length_unit="Mpc",
        mass_unit="Msun",
        bbox=bbox,
        periodicity=(False, False, False),
    )

    ds.sph_smoothing_style = "gather"  # seems to give more reliable results

    # set cosmology parameters

    ds.cosmological_simulation = 1
    ds.current_redshift = redshift
    ds.hubble_constant = 0.01 * cosmo.H0.value
    ds.omega_matter = cosmo.Om0
    ds.omega_lambda = cosmo.Ode0
    ds.omega_curvature = cosmo.Ok0
    ds.omega_radiation = cosmo.Onu0 + cosmo.Ogamma0

    if ("gas", "density") in ds.field_list:
        # if hydro sim, add derived fields

        # compute a new MMW field
        ds.add_field(
            ("gas", "MMW"),
            function=_mmw,
            units="",
            sampling_type="particle",
        )

        ds.add_field(
            ("gas", "temperature"),
            function=_temperature,
            units="K",
            sampling_type="particle",
        )

        ds.add_field(
            ("gas", "number_density"),
            function=_number_density,
            units="cm**-3",
            sampling_type="particle",
            force_override=True,
        )

        ds.add_field(
            ("gas", "xh"), function=_h_fraction, units="", sampling_type="particle"
        )

        ds.add_field(
            ("gas", "metallicity"),
            function=_metallicity,
            units="Zsun",
            sampling_type="particle",
        )

        if compute_xray_fields:
            # compute xray luminosities, emissivities, etc. using pyxsim.
            # This calls CIESourceModel, which assumes ionization equilibrium.
            # User can define custom parameters

            ds.add_field(
                ("gas", "emission_measure"),
                function=_emission_measure,
                units="cm**-3",
                sampling_type="particle",
            )

            default_kwargs = {
                "model": "apec",
                "emin": 0.1,  # keV
                "emax": 10.0,  # keV
                "nbins": 1000,
                "Zmet": ("gas", "metallicity"),  # Zsun
                "temperature_field": ("gas", "temperature"),
                "emission_measure_field": ("gas", "emission_measure"),
                "h_fraction": "xh",
            }

            if source_model_kwargs is None:
                source_model_kwargs = {}

            # update with user-defined settings
            source_model_kwargs = {**default_kwargs, **source_model_kwargs}

            # define xray source model (
            # NOTE: this will download a few fits files needed for the analysis)
            source = CIESourceModel(**source_model_kwargs)

            # populate yt dataset with xray fields
            source.make_source_fields(
                ds, source_model_kwargs["emin"], source_model_kwargs["emax"]
            )

            if return_source_model:
                return ds, source

    elif compute_xray_fields:
        raise RuntimeError(
            "`compute_xray_fields` can only be used with hydrodynamic simulations"
        )

    return ds


# ---------- DERIVED FIELDS -------------- #


def _mmw(field, data):
    # Recompute mean molecular weight. The "mu" field currently stored
    # is 1.0 with units of kg, which is wrong.
    Y = data["gas", "yhe"].d
    X = 1 - Y
    Z = data["gas", "zmet"].d * solar_metallicity

    # MMW for fully ionized gas
    return 1 / (2 * X + 0.75 * Y + Z / (2 * 16))


def _temperature(field, data):
    gamma = 5 / 3
    return (
        data["gas", "MMW"] * mp * data["gas", "uu"].to("cm**2/s**2") / kB * (gamma - 1)
    )


def _number_density(field, data):
    return data["gas", "density"].to("g/cm**3") / (data["gas", "MMW"] * mp)


def _metallicity(field, data):
    # metallicity in solar units
    return data["gas", "zmet"]


def _h_fraction(field, data):
    return 1 - data["gas", "yhe"]


def _emission_measure(field, data):
    # emission_measure = ne**2 * particle_volume

    # assume gas is fully ionized -- safe assumption for cluster scale objects
    ne = (1 - 0.5 * data["gas", "yhe"].d) * data["gas", "density"].to("g/cm**3") / mp
    nH = (1 - data["gas", "yhe"].d) * data["gas", "density"].to("g/cm**3") / mp

    return ne * nH * (data["gas", "particle_mass"] / data["gas", "density"]).to("cm**3")
