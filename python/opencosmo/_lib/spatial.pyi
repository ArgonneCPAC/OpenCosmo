import numpy as np
import numpy.typing as npt

def get_closest_distance_3d(
    check_vecs: npt.NDArray[np.float64],
    query_vecs: npt.NDArray[np.float64],
    max_distance: float,
) -> npt.NDArray[np.float64]: ...
