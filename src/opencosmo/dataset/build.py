from typing import Optional

import h5py

import opencosmo as oc
from opencosmo.dataset import Dataset
from opencosmo.dataset.handler import DatasetHandler
from opencosmo.dataset.state import DatasetState
from opencosmo.header import OpenCosmoHeader
from opencosmo.index import ChunkedIndex, DataIndex
from opencosmo.spatial.tree import open_tree
from opencosmo.transformations import units as u


def build_dataset(
    file: h5py.File | h5py.Group,
    header: OpenCosmoHeader,
    index: Optional[DataIndex] = None,
) -> Dataset:
    """
    Builds a dataset. Used when header and the data group are not
    in the same place
    """
    try:
        tree = open_tree(file, header.simulation.box_size)
    except ValueError:
        tree = None

    p1 = (0, 0, 0)
    p2 = tuple(header.simulation.box_size for _ in range(3))
    sim_box = oc.make_box(p1, p2)
    builders, base_unit_transformations = u.get_default_unit_transformations(
        file, header
    )

    handler = DatasetHandler(file)

    if index is None:
        index = ChunkedIndex.from_size(len(handler))
    state = DatasetState(
        base_unit_transformations,
        builders,
        index,
        u.UnitConvention.SCALEFREE,
        sim_box,
        header,
    )

    dataset = Dataset(
        handler,
        header,
        state,
        tree,
    )
    return dataset
