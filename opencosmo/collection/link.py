"""
Some types of data contain links to other data. In particular, property files
contain links to their particle files. HaloProperty files contain links to
halo particles, which include AGN, Dark Matter, Gas, and Star particles.
GalaxyProperties contain a link to the associated star particles in the 
GalaxyParticles file. A link is simply a combination of a starting index
and a length, which maps a single row in a property file to a range of rows
in a particle file.

As such, efficient querying of linking can only be done between specific types
of data
"""
from __future__ import annotations
from opencosmo.header import OpenCosmoHeader
from collections import defaultdict
import numpy as np
from typing import TypedDict
from h5py import File, Group


LINK_ALIASES = { # Name maps
    'sodbighaloparticles': 'halo_particles',
    'sod_profile': 'halo_profiles',
}

ALLOWED_LINKS = { # Files that can serve as a link holder and 
    'halo_properties': ["halo_particles", 'halo_profiles'],
    'galaxy_properties': ["galaxy_particles"]
}


def verify_links(*headers: OpenCosmoHeader) -> dict[str, list[str]]:
    """
    Verify that the links in the headers are valid. This means that the
    link holder has a corresponding link target and that the link target
    is of the correct type. It also verifies that the linked files are from
    the same simulation. Returns a dictionary where the keys are the
    link holder files and the values are lists of the corresponding link.

    Raises an error if the links are not valid, otherwise returns the links.
    """

    data_types = [header.file.data_type for header in headers]
    if len(set(data_types)) != len(data_types):
        raise ValueError("Data types in files must be unique to link correctly")

    master_files = [dt for dt in data_types if dt in ALLOWED_LINKS]
    if not master_files:
        raise ValueError("No valid link holder files found in headers")

    dtypes_to_headers = {header.file.data_type: header for header in headers}

    links = defaultdict(list)  # {file: [link_header, ...]}
    for file in master_files:
        for link in ALLOWED_LINKS[file]:
            try:
                link_header = dtypes_to_headers[link]
                # Check that the headers come from the same simulation
                if link_header.simulation != dtypes_to_headers[file].simulation:
                    raise ValueError(f"Simulation mismatch between {file} and {link}")
                links[file].append(link)
            except KeyError:
                continue  # No link header found for this file

    # Master files also need to have the same simulation
    if len(master_files) > 1:
        raise ValueError("Data linking can only have one master file")
    for file in master_files:
        if dtypes_to_headers[file].simulation != dtypes_to_headers[master_files[0]].simulation:
            raise ValueError(f"Simulation mismatch between {file} and {master_files[0]}")
    output_file = master_files[0]
    return output_file, links[output_file]

def get_links(file: File | Group) -> dict[str, np.ndarray]:
    if "data_linked" not in file.keys():
        raise ValueError(f"No links found in {file.name}")
    keys = file["data_linked"].keys()
    # Remove everything after the last underscore to get unique link names
    unique_keys = {key.rsplit('_', 1)[0] for key in keys}
    if any(k.startswith("sod") for k in unique_keys):
        # we're dealing with a halo property file
        return read_halo_property_links(file)
    else:
        # we're dealing with a galaxy property file
        size = file["data_linked"]["galaxyparticles_star_particles_size"][()]
        start = file["data_linked"]["galaxyparticles_star_particles_start"][()]
        return GalaxyPropertyLink({
            "galaxy_particles": {
                "start_index": start,
                "length": size
            }
        })
    
def read_halo_property_links(file: File | Group) -> HaloPropertyLink:
    """
    Read the links from a halo property file. The links are stored in the
    "data_linked" group of the file. Each link is a combination of a
    starting index and a length, which maps a single row in the property
    file to a range of rows in the particle file.
    """
    
    # Read the links for dark matter, AGN, gas, and star particles
    return HaloPropertyLink({
        "dm_particles": {
            "start_index": file["data_linked"]["sodbighaloparticles_dm_particles_start"][()],
            "length": file["data_linked"]["sodbighaloparticles_dm_particles_size"][()]
        },
        "agn_particles": {
            "start_index": file["data_linked"]["sodbighaloparticles_agn_particles_start"][()],
            "length": file["data_linked"]["sodbighaloparticles_agn_particles_size"][()]
        },
        "gas_particles": {
            "start_index": file["data_linked"]["sodbighaloparticles_gas_particles_start"][()],
            "length": file["data_linked"]["sodbighaloparticles_gas_particles_size"][()]
        },
        "star_particles": {
            "start_index": file["data_linked"]["sodbighaloparticles_star_particles_start"][()],
            "length": file["data_linked"]["sodbighaloparticles_star_particles_size"][()]
        },
        "galaxy_properties": {
            "start_index": file["data_linked"]["galaxyproperties_start"][()],
            "length": file["data_linked"]["galaxyproperties_size"][()]
        },
        "halo_profiles": file["data_linked"]["sod_profile_idx"][()]
    })

class DataLink(TypedDict):
    start_index: np.ndarray  # The starting index of the link in the particle file
    length: np.ndarray  # The

class HaloPropertyLink(TypedDict):            
    dm_particles: DataLink
    agn_particles: DataLink
    gas_particles: DataLink
    star_particles: DataLink
    galaxy_properties: DataLink
    halo_profiles: np.ndarray

class GalaxyPropertyLink(TypedDict):
    galaxy_particles: DataLink
