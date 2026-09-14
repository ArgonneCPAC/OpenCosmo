# contains general function for computing various statistics

import opencosmo as oc
import numpy as np

from opencosmo.mpi import get_comm_world, get_mpi
from opencosmo.analysis.default_plotting_params import default_params

MPI = get_mpi()
comm = get_comm_world()
rank = comm.Get_rank() if comm is not None else 0
ranks = comm.Get_size() if comm is not None else 1


def _get_statistic(col, statistic, **kwargs):
    # return either result of custom, callable input function or that of 
    # a builtin method attached to the Column object

    if isinstance(statistic, str):
        if statistic.startswith("geometric_"):
            # if doing geometric mean, median, quantile, etc., does
            # global scalar reduction in log space. Will need to
            # convert back to exp10() space after the function call.

            return getattr( col.log10(), statistic.removeprefix("geometric_") )(**kwargs)

        else:
            return getattr(col, statistic)(**kwargs)

    if callable(statistic):
        return statistic(col, **kwargs)

    raise TypeError("statistic must be a string or callable")

def _compute_bins(min, max, n, bin_spacing="log"):
    if bin_spacing == "log":
        return np.geomspace(min, max, n)
    elif bin_spacing == "linear":
        return np.linspace(min, max, n)
    else:
        raise RuntimeError(f"unrecognized value for bin_spacing: {bin_spacing}")

def _filter_bad(ds, column):

    params = default_params.get(column)
    if params is None:
        return ds

    filters = params.get("filter_bad")
    if filters is None:
        return ds

    if not isinstance(filters, (list, tuple)):
        filters = [filters]

    return ds.filter(*filters)


def binned_statistic(
    ds, column, 
    bin_by="sod_halo_mass", 
    statistic="mean", 
    bins=20, 
    dataset="halo_properties", 
    mode="global", 
    **kwargs,
):
    # statistic can be either string of a function that takes the column as input
    #   def cool_stat(col):
    #       return (col.max()-col.min()) / col.std()
    #
    #   stat = cool_stat(oc.col("concentration"))
    '''
    Computes a statistic binned by the `bin_by` column (e.g. mean halo concentration in bins of SOD halo mass).
    The statistic can be any of the pre-build scalar reductions (in string form -- e.g., "mean", "std"), or a custom
    statistic that takes the column as input. 
    '''


    if isinstance(ds, oc.StructureCollection):
        ds = ds[dataset]

    ds = _filter_bad(ds, column)

    if not isinstance(bins, list):

        d = ds.select( 
                bin_min = oc.col(bin_by).min(), 
                bin_max = oc.col(bin_by).max(),
                mode = mode,
            ).get_data()

        bins = np.geomspace(d["bin_min"], d["bin_max"], bins+1)

    # else:
    #   make sure given bins are in the right units

    binned_stat = []

    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]

        d = (
            ds.filter(oc.col(bin_by) >= low, oc.col(bin_by) < high)
            .select( 
                stat = _get_statistic(oc.col(column), statistic, **kwargs),
                mode = mode,
            )
            .get_data()
        )

        # convert back from log-space if doing geometric mean, median, etc.
        if isinstance(statistic, str) and statistic.startswith("geometric_"):
            d = 10 ** d

        binned_stat.append(d)

        if ranks > 1:
            comm.Barrier()

    return binned_stat, bins

def hist1d(
    ds,
    column, 
    bins=20,
    bin_spacing = "log",
    mode="global",
):

    if isinstance(ds, oc.StructureCollection):
        ds = ds[dataset]

    ds = _filter_bad(ds, column)

    if not isinstance(bins, list):

        d = ds.select( 
                bin_min = oc.col(column).min(), 
                bin_max = oc.col(column).max(),
                mode = mode,
            ).get_data()

        bins = _compute_bins(
            d["bin_min"], d["bin_max"], bins+1, 
            bin_spacing = bin_spacing
        )

    counts, bin_edges = np.histogram( ds.select(column).get_data(), bins=bins )

    if ranks > 1:
        counts = comm.allreduce( counts, op=MPI.SUM )

    return counts, bin_edges

def hist2d(
    ds,
    column_x,
    column_y, 
    bins=100,
    bin_spacing="log",
    mode="global",
):

    if isinstance(ds, oc.StructureCollection):
        ds = ds[dataset]

    ds = _filter_bad(ds, column_x)
    ds = _filter_bad(ds, column_y)

    if not isinstance(bins, list):
        d_x = ds.select( 
                bin_min = oc.col(column_x).min(), 
                bin_max = oc.col(column_x).max(),
                mode = mode,
            ).get_data()

        d_y = ds.select( 
                bin_min = oc.col(column_y).min(), 
                bin_max = oc.col(column_y).max(),
                mode = mode,
            ).get_data()

        if isinstance(bin_spacing, (list, tuple)):
            bin_spacing_x, bin_spacing_y = bin_spacing
        elif isinstance(bin_spacing, str):
            bin_spacing_x = bin_spacing
            bin_spacing_y = bin_spacing
        else:
            raise RuntimeError(f"Invalid input for bin_spacing: {bin_spacing}")
    
        bins_x = _compute_bins(
            d_x["bin_min"], d_x["bin_max"], bins+1, 
            bin_spacing = bin_spacing_x
        )

        bins_y = _compute_bins(
            d_y["bin_min"], d_y["bin_max"], bins+1, 
            bin_spacing = bin_spacing_y
        )

    else:
        bins_x, bins_y = bins

    data = ds.select(column_x, column_y).get_data()

    h, x, y = np.histogram2d(data[column_x], data[column_y], bins = [bins_x, bins_y])

    if ranks > 1:
        h = comm.allreduce( h, op=MPI.SUM )

    return h, x, y

def stacked_profile(
    ds,
    column,
    mode="global",
    statistic="mean",
    stat_space="log",
):

    if isinstance(ds, oc.StructureCollection):
        ds = ds["halo_profiles"]

    return





def two_point_correlation_function():
    return

def halo_mass_function():
    return