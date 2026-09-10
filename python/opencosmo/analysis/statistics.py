# contains general function for computing various statistics

import opencosmo as oc
import numpy as np

from opencosmo.mpi import get_comm_world, get_mpi

MPI = get_mpi()
comm = get_comm_world()
rank = comm.Get_rank() if comm is not None else 0
ranks = comm.Get_size() if comm is not None else 1


#NOTE: FOR PLOTTING, ADD A WAY TO PLOT FOR MULTIPLE SUBVOLUMES

def _pmin(x):
    if ranks > 1:
        return min( comm.allreduce(x, op=MPI.MIN) )
    else:
        return min(x)

def _pmax(x):
    if ranks > 1:
        return max( comm.allreduce(x, op=MPI.MAX) )
    else:
        return max(x)

def _pmean(x, w=None):
    if w is None:
        w = np.ones_like(x)

    if ranks > 1:
        num   = comm.allreduce(x * w, op=MPI.SUM)
        denom = comm.allreduce(w, op=MPI.SUM)

        if denom == 0:
            return 0
        else:
            return num/denom 

    else:
        return( sum(x*w)/sum(w) )


def _pmedian():
    return

def _get_statistic(col, statistic, **kwargs):

    if isinstance(statistic, str):
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


def binned_statistic(
    ds, column, 
    bin_by="sod_halo_mass", 
    statistic="mean", 
    bins=20, 
    dataset="halo_properties", 
    mode="global", 
    *kwargs,
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


    if not isinstance(bins, list):

        d = ds.select( 
                bin_min = oc.col(bin_by).min(), 
                bin_max = oc.col(bin_by).max(),
                mode = mode,
            ).get_data()

        bins = np.geomspace(d["bin_min"], d["bin_max"], bins+1)

    #print(f"BINS: {bins}", flush=True)

    # else:
    #   make sure given bins are in the right units

    binned_stat = []

    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]

        #print(f"[{low}, {high}]", flush=True)

        d = (
            ds.filter(oc.col(bin_by) >= low, oc.col(bin_by) < high)
            .select( 
                stat = _get_statistic(oc.col(column), statistic, **kwargs),
                mode = mode,
            )
            .get_data()
        )

        #print(f"{statistic}: {d}", flush=True)

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
    bins=20,
    bin_spacing="log",
    mode="global",
):

    if isinstance(ds, oc.StructureCollection):
        ds = ds[dataset]

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
):

    if isinstance(ds, oc.StructureCollection):
        ds = ds["halo_profiles"]

    return





def two_point_correlation_function():
    return

def halo_mass_function():
    return