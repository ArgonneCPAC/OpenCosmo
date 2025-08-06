from inspect import signature
from typing import Callable, NewType, Optional

import numpy as np
from numpy.typing import DTypeLike

import opencosmo as oc

StructureCollectionVisitorSpec = NewType(
    "StructureCollectionVisitorSpec", dict[str, Optional[set[str]]]
)


def visit_structure_collection(
    function: Callable,
    spec: StructureCollectionVisitorSpec,
    collection: oc.StructureCollection,
    dtype: Optional[DTypeLike] = None,
):
    __verify(function, spec, collection)
    to_visit = __prepare(spec, collection)
    if dtype is None:
        dtype = np.float64
    storage = np.zeros(len(collection), dtype=dtype)
    if isinstance(to_visit, oc.Dataset):
        raise NotImplementedError()

    for i, structure in enumerate(collection.objects()):
        storage[i] = function(**structure)

    return storage


def __prepare(
    spec: StructureCollectionVisitorSpec, collection: oc.StructureCollection
) -> oc.Dataset | oc.StructureCollection:
    if len(spec.keys()) == 1:
        ds_name = next(iter(spec.keys()))
        if ds_name.startswith(collection.dtype):
            ds = collection[ds_name]
            columns = spec[ds_name]

            if columns is not None:
                return ds.select(columns)
    collection = collection.with_datasets(list(spec.keys()))
    for ds_name, columns in spec.items():
        if columns is None:
            continue
        collection = collection.select(columns, dataset=ds_name)
    return collection


def __verify(
    function: Callable,
    spec: StructureCollectionVisitorSpec,
    collection: oc.StructureCollection,
):
    datasets_in_collection = set(collection.keys())
    datasets_in_spec = set(spec.keys())

    if not datasets_in_spec.issubset(datasets_in_collection):
        raise ValueError(
            "This collection is missing datasets "
            f"{datasets_in_spec - datasets_in_collection} requested for this visitor"
        )
    for ds_name, columns_in_spec in spec.items():
        if columns_in_spec is None:
            continue
        columns_in_spec = set(columns_in_spec)
        columns_in_dataset = collection[ds_name].columns
        if not columns_in_spec.issubset(columns_in_dataset):
            raise ValueError(
                "Dataset {ds_name} is missing columns "
                f"{columns_in_spec - columns_in_dataset} requested for this visitor"
            )

    fn_signature = signature(function)
    parameter_names = set(fn_signature.parameters.keys())
    if not datasets_in_spec.issubset(parameter_names):
        raise ValueError(
            "Visitor function must use the names of the datasets it requests as its "
            "argument names"
        )
