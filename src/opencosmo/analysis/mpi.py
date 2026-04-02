from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from opencosmo.io.mpi import get_all_keys
from opencosmo.mpi import MPI, get_comm_world

if TYPE_CHECKING:
    from collections.abc import Callable


class EvalOperation(Enum):
    SUM = "sum"
    PROD = "prod"
    AVG = "avg"


def reduce(
    dataset,
    function,
    operation: str = "sum",
    all: bool = False,
    plotting_function: Callable | None = None,
    evaluate_kwargs: dict[str, Any] | None = None,
    plotting_kwargs: dict[str, Any] | None = None,
    **ekwargs,
):
    r"""
    Combine results from several MPI processes into a single result. By defualt, the result is returned
    to the root process (rank 0), while all other processes are returned :code:`None`. You can
    return the result to all processes by setting :code:`all = True`.

    Under the hood, this function uses :py:meth:`evaluate <opencosmo.Dataset.evaluate>` to perform the
    computation. Besides the specific arguments mentioned below, you should pass in the arguments
    that you would if you were calling :code:`evaluate` directly (including :code:`vectorize`, which you will
    probably want to set to :code:`True`)

    If you like, you can also pass a plotting function


    For example, to compute a halo mass function across a large simulation:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import opencosmo as oc
        from opencosmo.analysis import reduce

        ds = oc.open("haloproperties.hdf5")
        def halo_mass_function(fof_halo_mass, log_bins, box_size):
            log_mass = np.log10(fof_halo_mass)
            hist, _ = np.histogram(log_mass, log_bins)
            return hist / np.diff(log_bins) / box_size**3


        def make_plot(halo_mass_function, log_bins, path, **kwargs):
            bin_centers = 0.5 * (log_bins[:-1] + log_bins[1:])
            plt.plot(bin_centers, halo_mass_function)
            plt.semilogy()
            plt.savefig(path)


        bins = np.linspace(10, 15)
        box_size = ds.header.simulation["box_size"].value
        plotting_arguments = {"path": "hmf.png"}
        evalute_kwargs = {"vectorize": True, "format": "numpy", "box_size": box_size, "log_bins": bins}

        reduce(
            ds,
            halo_mass_function,
            plotting_function=make_plot,
            evalute_kwargs=evalute_kwargs,
            plotting_kwargs=plotting_arguments,
        )

    When using this function, it's generally recommended you add a \*\*kwargs to your plotting function since it will recieve
    all of the evaulate keyword arugments in addition to the plotting arguments.

    This function checks that the values returned from the different processes can actually
    be combined, and throws an error if not. The most common failure cases is when the
    arrays returned by various processes are not the same size.

    Althoug the example above only returns a single array, you may return multiple
    arrays as a dictionary. Each array in the dictionary will be processed seperately.

    Parameters
    ----------
    dataset: Dataset | Collection
        Any OpenCosmo dataset or collection which supports :code:`evaluate`
    function: Callable
        A function to compute on the dataset. See the documentation for :code:`evaluate` for your
        given data type for details on the expected signature.
    operation: string, "sum" | "prod" | "avg", default = "sum"
        The operation to use when performing the reduction. If "avg", the averages will be weighted by the relative
        sizes of the datasets on each rank.
    all: bool, default = False
        Whether to return the result to all processes or just the root process. If :code:`False`, all processes besides
        the root process will recieve :code:`None`
    plotting_function: Optional[Callable], default = None
        A function that performs some plotting or post-processing. Since the result from `evaluate` function is always
        a dictionary, this function should take arguments with the same name as the keys of this dictionary. The :code:`evaluate_kwargs`
        will also be passed into this function as keyword arguments.

    plotting_kwargs: dict[str, Any]
        Additional keyword arguments to pass into the plotting function.

    **evaluate_kwargs: Any
        Additional keyword arguments that will be passed directly into :code:`dataset.evalute` and :code:`plotting_function` (if applicable)

    Returns
    -------
    results: dict[str, np.ndarray] | None
        The result of the reduction. If :code:`all = False` (the default) only the root process will recieve
        the results with the remaining processes receiving :code:`None`. If :code:`all = True`, all processes
        will recieve the results

    """
    evaluate_kwargs = evaluate_kwargs or {}
    plotting_kwargs = plotting_kwargs or {}

    _ = evaluate_kwargs.pop("insert", None)
    comm = get_comm_world()
    if comm is None:
        result = dataset.evaluate(function, insert=False, **evaluate_kwargs, **ekwargs)
        return process_output(
            result, plotting_function, plotting_kwargs, evaluate_kwargs | ekwargs
        )

    op = EvalOperation(operation)
    result = dataset.evaluate(function, insert=False, **evaluate_kwargs, **ekwargs)
    results_to_combine = __verify_results(result, comm)
    keys = get_all_keys(results_to_combine, comm)
    reduce_func = comm.allreduce if all else comm.reduce
    output = {}

    match op:
        case EvalOperation.AVG:
            total_size = comm.allreduce(len(dataset))
            weight = len(dataset) / total_size
            results_to_combine = {
                name: value * weight for name, value in results_to_combine.items()
            }
            combine_operation = MPI.SUM
        case EvalOperation.SUM:
            combine_operation = MPI.SUM
        case EvalOperation.PROD:
            combine_operation = MPI.PROD

    for key in keys:
        output[key] = reduce_func(results_to_combine[key], op=combine_operation)
    if not all and comm.Get_rank() != 0:
        return None

    if not isinstance(result, dict):
        return next(iter(output.values()))

    return process_output(output, plotting_function, plotting_kwargs, evaluate_kwargs)


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


def process_output(
    output: dict[str, np.ndarray],
    plotting_function: Callable | None,
    plotting_kwargs: dict[str, Any],
    evaluate_kwargs: dict[str, Any],
):
    if plotting_function is None:
        return output
    return plotting_function(**output, **plotting_kwargs, **evaluate_kwargs)
