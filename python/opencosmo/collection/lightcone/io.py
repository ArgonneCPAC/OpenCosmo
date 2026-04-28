from __future__ import annotations

from collections import OrderedDict, defaultdict

from opencosmo.collection.lightcone import utils as lcutils
from opencosmo.dataset import dataset as ocds
from opencosmo.io.mpi import get_all_keys
from opencosmo.mpi import get_comm_world, get_mpi


def order_by_redshift_range(datasets: dict[str, ocds.Dataset]):
    redshift_ranges = {
        key: lcutils.get_single_redshift_range(ds) for key, ds in datasets.items()
    }
    sorted_ranges = sorted(redshift_ranges.items(), key=lambda item: item[1][0])
    output = OrderedDict()
    for name, _ in sorted_ranges:
        output[name] = datasets[name]
    return output


def combine_adjacent_datasets_mpi(
    ordered_datasets: dict[str, dict[str, ocds.Dataset]],
    min_dataset_size,
):
    MIN_DATASET_SIZE = 100_000
    comm = get_comm_world()
    MPI = get_mpi()
    all_dataset_steps = get_all_keys(ordered_datasets, comm)
    assert comm is not None and MPI is not None
    rs = 0
    output_datasets: dict[str, list[dict[str, ocds.Dataset]]] = OrderedDict()
    for step in all_dataset_steps:
        if rs == 0:
            current_key = step
            output_datasets[current_key] = []

        if step not in ordered_datasets:
            rs += comm.allreduce(0, MPI.SUM)
        else:
            length = sum(len(ds) for ds in ordered_datasets[step].values())
            rs += comm.allreduce(length)
            output_datasets[current_key].append(ordered_datasets[step])

        if rs > MIN_DATASET_SIZE:
            rs = 0

    output = OrderedDict()
    for step, datasets in output_datasets.items():
        step_output = defaultdict(list)
        for ds_group in datasets:
            for ds_type, ds in ds_group.items():
                step_output[ds_type].append(ds)
        output[step] = step_output

    return output


def combine_adjacent_datasets(
    ordered_datasets: dict[str, ocds.Dataset] | dict[str, dict[str, ocds.Dataset]],
    min_dataset_size=100_000,
):
    is_single = isinstance(next(iter(ordered_datasets.values())), ocds.Dataset)
    datasets: dict[str, dict[str, ocds.Dataset]]
    if is_single:
        assert all(isinstance(ds, ocds.Dataset) for ds in ordered_datasets.values())
        datasets = {key: {"data": ds} for key, ds in ordered_datasets.items()}  # type: ignore
    else:
        assert all(isinstance(ds, dict) for ds in ordered_datasets.values())
        datasets = ordered_datasets  # type: ignore

    if get_comm_world() is not None:
        return combine_adjacent_datasets_mpi(datasets, min_dataset_size)

    running_sum = 0

    current_key = next(iter(ordered_datasets.keys()))
    output_datasets: dict[str, list[dict[str, ocds.Dataset]]] = OrderedDict(
        {current_key: []}
    )

    for key, step_datasets in datasets.items():
        if running_sum < min_dataset_size:
            running_sum += sum(len(ds) for ds in step_datasets.values())
            output_datasets[current_key].append(step_datasets)
            continue
        current_key = key
        output_datasets[current_key] = [step_datasets]
        running_sum = sum(len(ds) for ds in step_datasets.values())

    # We have list of dicts, go to dict of lists
    output = OrderedDict()
    for step, step_datasets_ in output_datasets.items():
        step_output = defaultdict(list)
        for ds_group in step_datasets_:
            for ds_type, ds in ds_group.items():
                step_output[ds_type].append(ds)
        output[step] = step_output

    return output
