from typing import Any

import astropy.units as u

import opencosmo as oc

default_params: dict[str, dict[str, Any]] = {
    "sod_halo_mass": {
        "filter_bad": oc.col("sod_halo_mass") > 0,
        "plotting": {
            "label": r"$M_\mathrm{200c}\,\,[M_\odot]$",
            "symbol": r"M_\mathrm{200c}",
            "scale": "log",
            "min": 1e12 * u.solMass,
            "max": 1e16 * u.solMass,
        },
    },
    "sod_halo_radius": {
        "filter_bad": oc.col("sod_halo_radius") > 0,
        "plotting": {
            "label": r"$R_\mathrm{200c}\,\,[\mathrm{Mpc}]$",
            "symbol": r"R_\mathrm{200c}",
            "scale": "log",
            "min": None,
            "max": None,
        },
    },
    "sod_halo_Y500c": {
        "filter_bad": oc.col("sod_halo_Y500c") > 0,
        "plotting": {
            "label": r"$Y_\mathrm{500c}\,\,[\mathrm{Mpc^2}]$",
            "symbol": r"Y_\mathrm{500c}",
            "scale": "log",
            "min": None,
            "max": None,
        },
    },
    "sod_halo_T500c": {
        "filter_bad": oc.col("sod_halo_T500c") > 0,
        "plotting": {
            "label": r"$T_\mathrm{500c}\,\,[\mathrm{K}]$",
            "symbol": r"T_\mathrm{500c}",
            "scale": "log",
            "min": None,
            "max": None,
        },
    },
    "sod_halo_core_entropy": {
        "filter_bad": oc.col("sod_halo_core_entropy") > 0,
        "plotting": {
            "label": r"$K_\mathrm{core}\,\,[\mathrm{keV\,cm^2}]$",
            "symbol": r"K_\mathrm{core}",
            "scale": "log",
            "min": None,
            "max": None,
        },
    },
    "sod_halo_cdelta": {
        "filter_bad": [
            oc.col("sod_halo_cdelta") > 0,
            oc.col("sod_halo_cdelta") < 100,
        ],
        "plotting": {
            "label": r"$c_\mathrm{200c}$",
            "symbol": r"c_\mathrm{200c}",
            "scale": "linear",
            "min": None,
            "max": None,
        },
    },
}