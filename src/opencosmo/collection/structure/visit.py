from inspect import signature
from typing import TYPE_CHECKING, Callable, Mapping, Optional

import numpy as np
from numpy.typing import DTypeLike

from opencosmo import dataset as ds
from opencosmo.visit import insert

if TYPE_CHECKING:
    from opencosmo import StructureCollection


def visit_structure_collection(
    function: Callable,
    spec: Mapping[str, Optional[list[str]]],
    collection: "StructureCollection",
    dtype: Optional[DTypeLike] = None,
):
    spec = dict(spec)
    __verify(function, spec, collection)
    to_visit = __prepare(spec, collection)
    if dtype is None:
        dtype = np.float64

    storage = __make_output(function, to_visit)

    if isinstance(to_visit, ds.Dataset):
        raise NotImplementedError()

    for i, structure in enumerate(to_visit.objects()):
        output = function(**structure)
        insert(storage, i, output)

    return storage


def __make_output(function: Callable, collection: "StructureCollection"):
    first_values = function(**next(collection.take(1, at="start").objects()))
    n = len(collection)
    if not isinstance(first_values, dict):
        name = function.__name__
        return {name: np.zeros(n, dtype=type(first_values))}
    else:
        return {
            name: np.zeros(n, dtype=type(val)) for name, val in first_values.items()
        }


def __prepare(spec: dict[str, Optional[list[str]]], collection: "StructureCollection"):
    if len(spec.keys()) == 1:
        ds_name = next(iter(spec.keys()))
        dataset = collection[ds_name]
        if isinstance(dataset, ds.Dataset):
            columns = spec[ds_name]
            if columns is not None:
                return dataset.select(columns)
            return dataset
        else:
            collection = collection.with_datasets(list(spec.keys()))
    for ds_name, columns in spec.items():
        if columns is None:
            continue
        collection = collection.select(columns, dataset=ds_name)
    return collection


def __verify(
    function: Callable,
    spec: dict[str, Optional[list[str]]],
    collection: "StructureCollection",
):
    datasets_in_collection = set(collection.keys())
    fn_signature = signature(function)
    parameter_names = set(fn_signature.parameters.keys())

    for name in parameter_names:
        if name not in spec:
            spec.update({name: None})

    datasets_in_spec = set(spec.keys())

    if not datasets_in_spec.issubset(datasets_in_collection):
        raise ValueError(
            "This collection is missing datasets "
            f"{datasets_in_spec - datasets_in_collection} requested for this visitor"
        )
    for ds_name, columns_in_spec in spec.items():
        if columns_in_spec is None:
            continue
        dataset = collection[ds_name]
        if not isinstance(dataset, ds.Dataset):
            raise NotImplementedError
        columns_to_check = set(columns_in_spec)
        columns_in_dataset = set(dataset.columns)
        if not columns_to_check.issubset(columns_in_dataset):
            raise ValueError(
                "Dataset {ds_name} is missing columns "
                f"{columns_to_check - columns_in_dataset} requested for this visitor"
            )

    if not datasets_in_spec.issubset(parameter_names):
        raise ValueError(
            "Visitor function must use the names of the datasets it requests as its "
            "argument names"
        )
