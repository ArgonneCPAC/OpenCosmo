from typing import Any
import opencosmo as oc

default_params: dict[str, dict[str, Any]] = {
    # friends-of-friends properties
    "fof_halo_mass": {
        "filter_bad": oc.col("fof_halo_mass") > 0,
        "plotting": {
            "label": r"$M_\mathrm{FOF}$",
            "symbol": r"M_\mathrm{FOF}",
            "scale": "log",
        },
    },
    "fof_halo_1D_vel_disp": {
        "filter_bad": oc.col("fof_halo_1D_vel_disp") > 0,
        "plotting": {
            "label": r"$\sigma_\mathrm{v,FOF}$",
            "symbol": r"\sigma_\mathrm{v,FOF}",
            "scale": "log",
        },
    },
    # spherical overdensity masses
    "sod_halo_mass": {
        "filter_bad": oc.col("sod_halo_mass") > 0,
        "plotting": {
            "label": r"$M_\mathrm{200c}$",
            "symbol": r"M_\mathrm{200c}",
            "scale": "log",
        },
    },
    "sod_halo_M200m": {
        "filter_bad": oc.col("sod_halo_M200m") > 0,
        "plotting": {
            "label": r"$M_\mathrm{200m}$",
            "symbol": r"M_\mathrm{200m}",
            "scale": "log",
        },
    },
    "sod_halo_M500c": {
        "filter_bad": oc.col("sod_halo_M500c") > 0,
        "plotting": {
            "label": r"$M_\mathrm{500c}$",
            "symbol": r"M_\mathrm{500c}",
            "scale": "log",
        },
    },
    "sod_halo_MVir": {
        "filter_bad": oc.col("sod_halo_MVir") > 0,
        "plotting": {
            "label": r"$M_\mathrm{vir}$",
            "symbol": r"M_\mathrm{vir}",
            "scale": "log",
        },
    },
    # spherical overdensity radii
    "sod_halo_radius": {
        "filter_bad": oc.col("sod_halo_radius") > 0,
        "plotting": {
            "label": r"$R_\mathrm{200c}$",
            "symbol": r"R_\mathrm{200c}",
            "scale": "log",
        },
    },
    "sod_halo_R200m": {
        "filter_bad": oc.col("sod_halo_R200m") > 0,
        "plotting": {
            "label": r"$R_\mathrm{200m}$",
            "symbol": r"R_\mathrm{200m}",
            "scale": "log",
        },
    },
    "sod_halo_R500c": {
        "filter_bad": oc.col("sod_halo_R500c") > 0,
        "plotting": {
            "label": r"$R_\mathrm{500c}$",
            "symbol": r"R_\mathrm{500c}",
            "scale": "log",
        },
    },
    "sod_halo_RVir": {
        "filter_bad": oc.col("sod_halo_RVir") > 0,
        "plotting": {
            "label": r"$R_\mathrm{vir}$",
            "symbol": r"R_\mathrm{vir}",
            "scale": "log",
        },
    },
    # mass by component, within R200c
    "sod_halo_mass_dm": {
        "filter_bad": oc.col("sod_halo_mass_dm") > 0,
        "plotting": {
            "label": r"$M_\mathrm{DM,200c}$",
            "symbol": r"M_\mathrm{DM,200c}",
            "scale": "log",
        },
    },
    "sod_halo_mass_gas": {
        "filter_bad": oc.col("sod_halo_mass_gas") > 0,
        "plotting": {
            "label": r"$M_\mathrm{gas,200c}$",
            "symbol": r"M_\mathrm{gas,200c}",
            "scale": "log",
        },
    },
    "sod_halo_mass_star": {
        "filter_bad": oc.col("sod_halo_mass_star") > 0,
        "plotting": {
            "label": r"$M_\mathrm{*,200c}$",
            "symbol": r"M_\mathrm{*,200c}",
            "scale": "log",
        },
    },
    "sod_halo_mass_agn": {
        "filter_bad": oc.col("sod_halo_mass_agn") > 0,
        "plotting": {
            "label": r"$M_\mathrm{AGN,200c}$",
            "symbol": r"M_\mathrm{AGN,200c}",
            "scale": "log",
        },
    },
    "sod_halo_mass_sfgas": {
        "filter_bad": oc.col("sod_halo_mass_sfgas") > 0,
        "plotting": {
            "label": r"$M_\mathrm{SFgas,200c}$",
            "symbol": r"M_\mathrm{SFgas,200c}",
            "scale": "log",
        },
    },
    "sod_halo_mass_wind": {
        "filter_bad": oc.col("sod_halo_mass_wind") > 0,
        "plotting": {
            "label": r"$M_\mathrm{wind,200c}$",
            "symbol": r"M_\mathrm{wind,200c}",
            "scale": "log",
        },
    },
    "sod_halo_mass_nHI": {
        "filter_bad": oc.col("sod_halo_mass_nHI") > 0,
        "plotting": {
            "label": r"$M_\mathrm{HI,200c}$",
            "symbol": r"M_\mathrm{HI,200c}",
            "scale": "log",
        },
    },
    "sod_halo_mmagn_mass": {
        "filter_bad": oc.col("sod_halo_mmagn_mass") > 0,
        "plotting": {
            "label": r"$M_\mathrm{AGN,max}$",
            "symbol": r"M_\mathrm{AGN,max}",
            "scale": "log",
        },
    },
    # gas mass in other apertures
    "sod_halo_MGas200m": {
        "filter_bad": oc.col("sod_halo_MGas200m") > 0,
        "plotting": {
            "label": r"$M_\mathrm{gas,200m}$",
            "symbol": r"M_\mathrm{gas,200m}",
            "scale": "log",
        },
    },
    "sod_halo_MGas500c": {
        "filter_bad": oc.col("sod_halo_MGas500c") > 0,
        "plotting": {
            "label": r"$M_\mathrm{gas,500c}$",
            "symbol": r"M_\mathrm{gas,500c}",
            "scale": "log",
        },
    },
    "sod_halo_MGas2500c": {
        "filter_bad": oc.col("sod_halo_MGas2500c") > 0,
        "plotting": {
            "label": r"$M_\mathrm{gas,2500c}$",
            "symbol": r"M_\mathrm{gas,2500c}",
            "scale": "log",
        },
    },
    "sod_halo_MGasVir": {
        "filter_bad": oc.col("sod_halo_MGasVir") > 0,
        "plotting": {
            "label": r"$M_\mathrm{gas,vir}$",
            "symbol": r"M_\mathrm{gas,vir}",
            "scale": "log",
        },
    },
    "sod_halo_MGasHot500c": {
        "filter_bad": oc.col("sod_halo_MGasHot500c") > 0,
        "plotting": {
            "label": r"$M_\mathrm{hot\,gas,500c}$",
            "symbol": r"M_\mathrm{hot\,gas,500c}",
            "scale": "log",
        },
    },
    # stellar mass in other apertures
    "sod_halo_MStar200m": {
        "filter_bad": oc.col("sod_halo_MStar200m") > 0,
        "plotting": {
            "label": r"$M_{*,\mathrm{200m}}$",
            "symbol": r"M_{*,\mathrm{200m}}",
            "scale": "log",
        },
    },
    "sod_halo_MStar500c": {
        "filter_bad": oc.col("sod_halo_MStar500c") > 0,
        "plotting": {
            "label": r"$M_{*,\mathrm{500c}}$",
            "symbol": r"M_{*,\mathrm{500c}}",
            "scale": "log",
        },
    },
    "sod_halo_MStar2500c": {
        "filter_bad": oc.col("sod_halo_MStar2500c") > 0,
        "plotting": {
            "label": r"$M_{*,\mathrm{2500c}}$",
            "symbol": r"M_{*,\mathrm{2500c}}",
            "scale": "log",
        },
    },
    "sod_halo_MStarVir": {
        "filter_bad": oc.col("sod_halo_MStarVir") > 0,
        "plotting": {
            "label": r"$M_{*,\mathrm{vir}}$",
            "symbol": r"M_{*,\mathrm{vir}}",
            "scale": "log",
        },
    },
    # baryon fractions and ratios (dimensionless)
    "sod_halo_GasFracShell2500c": {
        "filter_bad": oc.col("sod_halo_GasFracShell2500c") > 0,
        "plotting": {
            "label": r"$f_\mathrm{gas,2500c}$",
            "symbol": r"f_\mathrm{gas,2500c}",
            "scale": "linear",
        },
    },
    "sod_halo_bhr": {
        "filter_bad": oc.col("sod_halo_bhr") > 0,
        "plotting": {
            "label": r"$\dot{M}_\mathrm{BH}$",
            "symbol": r"\dot{M}_\mathrm{BH}",
            "scale": "linear",
        },
    },
    # concentration and profile shape
    "sod_halo_cdelta": {
        "filter_bad": [
            oc.col("sod_halo_cdelta") > 0,
            oc.col("sod_halo_cdelta") < 100,
        ],
        "plotting": {
            "label": r"$c_\mathrm{200c}$",
            "symbol": r"c_\mathrm{200c}",
            "scale": "linear",
        },
    },
    "sod_halo_cdelta_error": {
        "filter_bad": oc.col("sod_halo_cdelta_error") > 0,
        "plotting": {
            "label": r"$\sigma_{c_\mathrm{200c}}$",
            "symbol": r"\sigma_{c_\mathrm{200c}}",
            "scale": "log",
        },
    },
    "sod_halo_c_acc_mass": {
        "filter_bad": [
            oc.col("sod_halo_c_acc_mass") > 0,
            oc.col("sod_halo_c_acc_mass") < 100,
        ],
        "plotting": {
            "label": r"$c_\mathrm{200c,acc}$",
            "symbol": r"c_\mathrm{200c,acc}",
            "scale": "linear",
        },
    },
    "sod_halo_c_peak_mass": {
        "filter_bad": [
            oc.col("sod_halo_c_peak_mass") > 0,
            oc.col("sod_halo_c_peak_mass") < 100,
        ],
        "plotting": {
            "label": r"$c_\mathrm{200c,peak}$",
            "symbol": r"c_\mathrm{200c,peak}",
            "scale": "linear",
        },
    },
    # kinematics
    "sod_halo_1D_vel_disp": {
        "filter_bad": oc.col("sod_halo_1D_vel_disp") > 0,
        "plotting": {
            "label": r"$\sigma_\mathrm{v,200c}$",
            "symbol": r"\sigma_\mathrm{v,200c}",
            "scale": "log",
        },
    },
    # temperatures
    "sod_halo_T500c": {
        "filter_bad": oc.col("sod_halo_T500c") > 0,
        "plotting": {
            "label": r"$T_\mathrm{500c}$",
            "symbol": r"T_\mathrm{500c}",
            "scale": "log",
        },
    },
    "sod_halo_T500cBolo": {
        "filter_bad": oc.col("sod_halo_T500cBolo") > 0,
        "plotting": {
            "label": r"$T_\mathrm{500c}^\mathrm{bolo}$",
            "symbol": r"T_\mathrm{500c}^\mathrm{bolo}",
            "scale": "log",
        },
    },
    "sod_halo_T500cBoloEx": {
        "filter_bad": oc.col("sod_halo_T500cBoloEx") > 0,
        "plotting": {
            "label": r"$T_\mathrm{500c}^\mathrm{bolo,ex}$",
            "symbol": r"T_\mathrm{500c}^\mathrm{bolo,ex}",
            "scale": "log",
        },
    },
    # integrated SZ signal
    "sod_halo_Y500c": {
        "filter_bad": oc.col("sod_halo_Y500c") > 0,
        "plotting": {
            "label": r"$Y_\mathrm{500c}$",
            "symbol": r"Y_\mathrm{500c}",
            "scale": "log",
        },
    },
    "sod_halo_Y5R500c": {
        "filter_bad": oc.col("sod_halo_Y5R500c") > 0,
        "plotting": {
            "label": r"$Y_\mathrm{5R500c}$",
            "symbol": r"Y_\mathrm{5R500c}",
            "scale": "log",
        },
    },
    # X-ray luminosities. these are stored as log10(erg/s) (see KNOWN_UNITS in
    # opencosmo/units/get.py), so the values are already logarithmic: the default axis
    # scale is linear
    "sod_halo_L500cBolo": {
        "filter_bad": None,
        "plotting": {
            "label": r"$L_\mathrm{500c}^\mathrm{bolo}$",
            "symbol": r"L_\mathrm{500c}^\mathrm{bolo}",
            "scale": "linear",
        },
    },
    "sod_halo_L500cBoloEx": {
        "filter_bad": None,
        "plotting": {
            "label": r"$L_\mathrm{500c}^\mathrm{bolo,ex}$",
            "symbol": r"L_\mathrm{500c}^\mathrm{bolo,ex}",
            "scale": "linear",
        },
    },
    "sod_halo_L500cErositaLo": {
        "filter_bad": None,
        "plotting": {
            "label": r"$L_\mathrm{500c}^\mathrm{eROSITA,lo}$",
            "symbol": r"L_\mathrm{500c}^\mathrm{eROSITA,lo}",
            "scale": "linear",
        },
    },
    "sod_halo_L500cErositaLoEx": {
        "filter_bad": None,
        "plotting": {
            "label": r"$L_\mathrm{500c}^\mathrm{eROSITA,lo,ex}$",
            "symbol": r"L_\mathrm{500c}^\mathrm{eROSITA,lo,ex}",
            "scale": "linear",
        },
    },
    "sod_halo_L500cErositaHi": {
        "filter_bad": None,
        "plotting": {
            "label": r"$L_\mathrm{500c}^\mathrm{eROSITA,hi}$",
            "symbol": r"L_\mathrm{500c}^\mathrm{eROSITA,hi}",
            "scale": "linear",
        },
    },
    "sod_halo_L500cErositaHiEx": {
        "filter_bad": None,
        "plotting": {
            "label": r"$L_\mathrm{500c}^\mathrm{eROSITA,hi,ex}$",
            "symbol": r"L_\mathrm{500c}^\mathrm{eROSITA,hi,ex}",
            "scale": "linear",
        },
    },
    "sod_halo_L500cRosat": {
        "filter_bad": None,
        "plotting": {
            "label": r"$L_\mathrm{500c}^\mathrm{ROSAT}$",
            "symbol": r"L_\mathrm{500c}^\mathrm{ROSAT}",
            "scale": "linear",
        },
    },
    "sod_halo_L500cRosatEx": {
        "filter_bad": None,
        "plotting": {
            "label": r"$L_\mathrm{500c}^\mathrm{ROSAT,ex}$",
            "symbol": r"L_\mathrm{500c}^\mathrm{ROSAT,ex}",
            "scale": "linear",
        },
    },
    # core / ICM properties
    "sod_halo_core_entropy": {
        "filter_bad": oc.col("sod_halo_core_entropy") > 0,
        "plotting": {
            "label": r"$K_\mathrm{core}$",
            "symbol": r"K_\mathrm{core}",
            "scale": "log",
        },
    },
    "sod_halo_core_ne": {
        "filter_bad": oc.col("sod_halo_core_ne") > 0,
        "plotting": {
            "label": r"$n_\mathrm{e,core}$",
            "symbol": r"n_\mathrm{e,core}",
            "scale": "log",
        },
    },
    "sod_halo_core_tcool": {
        "filter_bad": oc.col("sod_halo_core_tcool") > 0,
        "plotting": {
            "label": r"$t_\mathrm{cool,core}$",
            "symbol": r"t_\mathrm{cool,core}",
            "scale": "log",
        },
    },
    # star formation
    "sod_halo_sfr": {
        "filter_bad": oc.col("sod_halo_sfr") > 0,
        "plotting": {
            "label": r"$\mathrm{SFR}$",
            "symbol": r"\mathrm{SFR}",
            "scale": "log",
        },
    },
}
