from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

from opencosmo.collection.lightcone.state import redshift_range
from opencosmo.mpi import get_all_entries, get_comm_world, get_mpi

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opencosmo.collection.lightcone.state import LeafKey
    from opencosmo.dataset.state import DatasetState

    StepDatasets = dict[int | str, dict[str, DatasetState]]


def order_by_redshift_range(
    leaves: Iterable[tuple[LeafKey, DatasetState]],
) -> StepDatasets:
    """
    Group leaves by step, ordered by the lower edge of each step's redshift
    range. Within a step, leaves are named by their subtype, or "data" if the
    lightcone has one dataset per step.
    """
    steps: StepDatasets = defaultdict(dict)
    for key, leaf in leaves:
        steps[key[0]][str(key[1]) if len(key) > 1 else "data"] = leaf
    return OrderedDict(
        sorted(
            steps.items(),
            key=lambda item: redshift_range(item[1].values())[0],
        )
    )


def combine_adjacent_datasets_mpi(
    ordered_datasets: StepDatasets,
    min_dataset_size: int,
    no_stack: bool,
):
    MIN_DATASET_SIZE = 100_000
    comm = get_comm_world()
    MPI = get_mpi()
    assert comm is not None and MPI is not None
    rs = 0
    output_datasets: dict[int | str, list[dict[str, DatasetState]]] = OrderedDict()
    for step, step_datasets in get_all_entries(ordered_datasets, comm):
        if rs == 0:
            current_key = step
            output_datasets[current_key] = []

        if step_datasets is None:
            rs += comm.allreduce(0, MPI.SUM)
        else:
            length = sum(len(ds) for ds in step_datasets.values())
            rs += comm.allreduce(length)
            output_datasets[current_key].append(step_datasets)

        if rs > MIN_DATASET_SIZE or no_stack:
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
    ordered_datasets: StepDatasets,
    min_dataset_size: int,
    no_stack: bool,
):
    if get_comm_world() is not None:
        return combine_adjacent_datasets_mpi(
            ordered_datasets, min_dataset_size, no_stack
        )

    running_sum = 0

    current_key = next(iter(ordered_datasets.keys()))
    output_datasets: dict[int | str, list[dict[str, DatasetState]]] = OrderedDict(
        {current_key: []}
    )

    for key, step_datasets in ordered_datasets.items():
        if not no_stack and running_sum < min_dataset_size:
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
