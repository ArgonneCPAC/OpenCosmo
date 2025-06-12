import numpy as np
import yt  # type: ignore
from unyt import unyt_quantity # type: ignore
from astropy.table import Table  # type: ignore
from yt.data_objects.static_output import Dataset as YT_Dataset  # type: ignore
from yt.visualization.plot_window import PlotWindow # type: ignore
from re import sub
from pyxsim import CIESourceModel # type: ignore
from typing import Optional, Dict, Any



# ---- define some constants ---- #
mp = unyt_quantity(1.67262192595e-24, 'g')
kB = unyt_quantity(1.380649e-16, 'erg/K')
solar_metallicity = 0.012899 # value used internally in HACC
# ------------------------------- #


def create_yt_dataset(
    data: Table, 
    compute_xray_fields: Optional[bool] = None,
    return_source_model: Optional[bool] = None,
    source_model_kwargs: Optional[Dict[str, Any]] = {}
) -> YT_Dataset:

    data_dict = {}
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
        "uu": "internal_energy",
    }

    # TODO: just use val.from_astropy instead of manually converting unit string?
    
    def yt_unit(unit):
        """Converts Astropy units to yt-compatible units."""
        if unit is None:
            return "dimensionless"

        unit = str(unit).replace("solMass", "Msun").replace(" / ", "/").replace(" ", "*")
        return sub(r"(\d+)", r"**\1", unit)

    for ptype in data.keys():
        if "particles" not in ptype:
            continue

        particle_data = data[ptype].data
        ptype_short = ptype.split("_")[0]

        for field in particle_data.keys():
            yt_field_name = special_fields.get(field, field)
            data_dict[(ptype_short, yt_field_name)] = (particle_data[field], yt_unit(particle_data[field].unit))

        minx, maxx = min(minx, min(particle_data['x'])), max(maxx, max(particle_data['x']))
        miny, maxy = min(miny, min(particle_data['y'])), max(maxy, max(particle_data['y']))
        minz, maxz = min(minz, min(particle_data['z'])), max(maxz, max(particle_data['z']))

    bbox = [[minx, maxx], [miny, maxy], [minz, maxz]]

    # Raise error if any bound is still infinite
    if any(np.isinf(bound) for axis in bbox for bound in axis):
        raise ValueError("Bounding box coordinates contain infinite values."
                         "Check input data for missing or invalid positions.")

    ds = yt.load_particles(
        data_dict,
        length_unit="Mpc",
        mass_unit="Msun",
        bbox=bbox
    )

    # add derived fields

    # compute a new MMW field (TODO: find better solution)
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
        sampling_type="particle"
    )

    ds.add_field(
        ("gas", "number_density"),
        function=_number_density,
        units="cm**-3",
        sampling_type="particle",
        force_override=True
    )

    ds.add_field(
        ("gas", "xh"),
        function=_h_fraction,
        units="",
        sampling_type="particle"
    )

    ds.add_field(
        ("gas", "metallicity"),
        function=_metallicity,
        units="Zsun",
        sampling_type="particle"
    )

    if compute_xray_fields:
        # compute xray luminosities, emissivities, etc. using pyxsim.
        # This calls CIESourceModel, which assumes ionization equilibrium.
        # User can define custom parameters 

        ds.add_field(
            ("gas", "emission_measure"),
            function=_emission_measure,
            units="cm**-3",
            sampling_type="particle"
        )

        #TODO: Make sure redshift is passed into this list
        default_kwargs = {
            "model": "apec",
            "emin": 0.1,  # keV
            "emax": 10.0,  # keV
            "nbins": 1000,
            "Zmet": ("gas","metallicity"), # Zsun
            "temperature_field": ("gas", "temperature"),
            "emission_measure_field": ("gas", "emission_measure"),
            "h_fraction": "xh",
        }

        if source_model_kwargs is None:
            source_model_kwargs = {}
        
        # update with user-defined settings
        source_model_kwargs = {**default_kwargs, **source_model_kwargs}

        # define xray source model (NOTE: this will download a few fits files needed for the analysis)
        source = CIESourceModel(**source_model_kwargs)

        # populate yt dataset with xray fields
        source.make_source_fields(ds, source_model_kwargs["emin"], source_model_kwargs["emax"])

        if return_source_model:
            return ds, source

    return ds

def particle_projection_plot(*args, **kwargs) -> PlotWindow:
    return yt.ParticleProjectionPlot(*args, **kwargs)

def profile_plot(*args, **kwargs) -> PlotWindow:
    return yt.ProfilePlot(*args, **kwargs)

def phase_plot(*args, **kwargs) -> PlotWindow:
    return yt.PhasePlot(*args, **kwargs)



# ---------- DERIVED FIELDS -------------- #

def _mmw(field, data):
    # Recompute mean molecular weight. The "mu" field currently stored
    # is 1.0 with units of kg, which is wrong.
    Y = data['gas','yhe'].d
    X = 1-Y
    Z = data['gas','zmet'].d * solar_metallicity

    # MMW for fully ionized gas
    return 1 / (2*X + 0.75*Y + Z/(2*16))

def _temperature(field, data):
    gamma = 5/3
    return data['gas','MMW'] * mp * data['gas','internal_energy'].to('cm**2/s**2') / kB * (gamma-1)

def _number_density(field, data):
    return data['gas','density'].to('g/cm**3') / (data['gas','MMW']*mp)

def _metallicity(field, data):
    # metallicity in solar units
    return data["gas","zmet"]

def _h_fraction(field, data):
    return 1 - data['gas','yhe']

def _emission_measure(field, data):
    # emission_measure = ne**2 * particle_volume

    # assume gas is fully ionized -- safe assumption for cluster scale objects
    ne = (1-0.5*data['gas','yhe'].d)*data['gas','density'].to('g/cm**3') / mp
    nH = (1-data['gas','yhe'].d)*data['gas','density'].to('g/cm**3') / mp

    return ne*nH * (data['gas','particle_mass']/data['gas','density']).to('cm**3')
    
