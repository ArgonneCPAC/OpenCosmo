from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from opencosmo.io.mpi import get_all_keys
from opencosmo.mpi import get_comm_world

if TYPE_CHECKING:
    from opencosmo.mpi import MPI


def reduce(
    dataset, function, operation: str = "sum", all: bool = False, **evaluate_kwargs
):
    """
    Combine results from several MPI processes into a single result. By defualt, the result is returned
    to the root process (rank 0), while all other processes are returned :code:`None`. You can
    return the result to all processes by setting :code:`all = True`.

    Under the hood, this function uses :py:meth:`evaluate <opencosmo.Dataset.evaluate>` to perform the
    computation. Because of this, the specific set of required arguments will depend on the type of the
    dataset or collection you are doing the computation on.

    For example, to compute a halo mass function across a large simulation:

    .. code-block:: python
        import matplotlib.pyplot as plt
        import numpy as np
        import opencosmo as oc
        from opencosmo.analysis import reduce

        ds = oc.open("haloproperties.hdf5")

        def halo_mass_function(fof_halo_mass, log_bins):
            log_mass = np.log(fof_halo_mass)
            hist, _ = np.histogram(log_mass, log_bins)
            return hist / np.diff(log_mass)

        bins = np.logspace(10, 15, 100)
        histogram = reduce(ds, halo_mass_function, log_bins = bins)
        if histogram is not None:
            plt.plot(bins, histogram)
            plt.savefig("hmf.png")

    If you call this function but you are not working with MPI, it will succeed but will
    print a warning.

    """
    _ = evaluate_kwargs.pop("insert", None)
    comm = get_comm_world()
    if comm is None:
        warn(
            "reduce was called could not get an MPI communicator. Either you are not running with MPI or mpi4py is not installed"
        )
        return dataset.evaluate(function, insert=False, **evaluate_kwargs)

    result = dataset.evaluate(function, insert=False, **evaluate_kwargs)
    results_to_combine = __verify_results(result, comm)
    keys = get_all_keys(results_to_combine, comm)
    reduce_func = comm.allreduce if all else comm.reduce

    output = {}

    for key in keys:
        output[key] = reduce_func(results_to_combine[key])

    if not all and comm.Get_rank() != 0:
        return None

    if not isinstance(result, dict):
        return next(iter(output.values()))
    return output


def __verify_results(result: dict[str, np.ndarray] | np.ndarray, comm: MPI.Comm):
    if not isinstance(result, dict):
        result_to_check = {"output": result}
    else:
        result_to_check = result

    keys = get_all_keys(result_to_check, comm)
    for key in keys:
        has_key = comm.allgather(key in result_to_check)
        if not all(has_key):
            raise ValueError("Not all processes got the same output!")
        is_arr = comm.allgather(isinstance(result_to_check[key], np.ndarray))
        if not all(is_arr):
            raise ValueError(
                "Reduce expects the returned results to be a numpy array or dictionary of numpy arrays"
            )

        lengths = set(comm.allgather(len(result_to_check[key])))
        if len(lengths) > 1:
            raise ValueError(
                "To reduce a result, outputs from all processes must be the same length!"
            )
    return result_to_check
