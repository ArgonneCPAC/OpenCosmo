import numpy as np
import yt  # type: ignore
from yt.units import Msun, Mpc, h  # type: ignore
from astropy.table import Table  # type: ignore
from yt.data_objects.static_output import Dataset as YT_Dataset  # type: ignore
import re

def create_yt_dataset(data: Table) -> YT_Dataset:
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
        "zmet": "metal_fraction"
    }

    def yt_unit(unit):
        """Converts Astropy units to yt-compatible units."""
        if unit is None:
            return "dimensionless"

        unit = str(unit).replace("solMass", "Msun").replace(" / ", "/").replace(" ", "*")
        return re.sub(r"(\d+)", r"**\1", unit)

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
        length_unit=1.0 * Mpc,
        mass_unit=1.0 * Msun,
        bbox=bbox
    )

    return ds
