import numpy as np
import numpy.typing as npt

from opencosmo.index import IndexArray

def get_closest_distance_3d(
    check_vecs: npt.NDArray[np.float64],
    query_vecs: npt.NDArray[np.float64],
    max_distance: float,
) -> npt.NDArray[np.float64]: ...
def partition_bounding_box(
    box_indices: IndexArray, box_size: float, level: int
) -> tuple[float, float, float, float, float, float]: ...
def get_octree_indices(
    box_size: float,
    query_bounds: tuple[float, float, float, float, float, float],
    max_level: int,
) -> list[tuple[IndexArray, IndexArray]]: ...
