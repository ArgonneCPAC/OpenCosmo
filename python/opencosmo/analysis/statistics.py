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


def binned_statistic(ds, column, bin_by="sod_halo_mass", statistic="mean", bins=20, dataset="halo_properties", mode="global", **kwargs):
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

    print(f"BINS: {bins}", flush=True)

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

        print(f"{statistic}: {d}", flush=True)

        binned_stat.append(d)

        if ranks > 1:
            comm.Barrier()

    return binned_stat

def hist1d(ds):
    return #reduce(ds, _hist1d, evaluate_kwargs=evaluate_kwargs)

def hist2d():
    return 