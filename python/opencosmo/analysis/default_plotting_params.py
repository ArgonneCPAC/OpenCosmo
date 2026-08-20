import astropy.units as u

default_plotting_params = {
    "sod_halo_mass" : {
        "label" : r"$M_\mathrm{200c}\,\,[M_\odot]$",
        "scale" : "log",
        "min" : 1e12 * u.solMass,
        "max" : 1e16 * u.solMass,
        "filter_bad" : oc.col("sod_halo_mass") > 0,
    },
    "sod_halo_radius" : {
        "xlabel" : r"$R_\mathrm{200c}\,\,[\mathrm{Mpc}]$",
        "xscale" : "log",
        "min" : None,
        "max" : None,
        "filter_bad" : oc.col("sod_halo_radius") > 0,
    },
    "sod_halo_Y500c" : {
        "xlabel" : r"$Y_\mathrm{500c}\,\,[\mathrm{Mpc^2}]$",
        "xscale" : "log",
        "min" : None,
        "max" : None,
        "filter_bad" : oc.col("sod_halo_Y500c") > 0,
    },
    "sod_halo_T500c" : {
        "xlabel" : r"$T_\mathrm{500c}\,\,[\mathrm{K}]$",
        "xscale" : "log",
        "min" : None,
        "max" : None,
        "filter_bad" : oc.col("sod_halo_T500c") > 0,
    },
    "sod_halo_core_entropy" : {
        "xlabel" : r"$K_\mathrm{core}\,\,[\mathrm{keV\,cm^2}]$",
        "xscale" : "log",
        "min" : None,
        "max" : None,
        "filter_bad" : oc.col("sod_halo_core_entropy") > 0,
    },
    "sod_halo_cdelta" : {
        "ylabel" : r"$C_\mathrm{200c}$",
        "yscale" : "linear",
        "min" : None,
        "max" : None,
        "filter_bad" : [oc.col("sod_halo_cdelta") > 0, oc.col("sod_halo_cdelta") < 1e3],
    },
}