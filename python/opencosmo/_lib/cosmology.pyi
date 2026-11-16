import numpy as np
import numpy.typing as npt

class Cosmology:
    def comoving_distance(self, z1: float, z2: float | None) -> float: ...
    def comoving_distances(
        self, redshifts: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def comoving_distance_grid(
        self, z_min: float, z_max: float, n: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def redshift_at_distance(self, distance: float) -> float: ...

def make_cosmology(
    h: float,
    Om0: float,
    Ode0: float | None = None,
    Ok0: float | None = None,
    Or0: float | None = None,
    w0: float | None = None,
    wa: float | None = None,
) -> Cosmology: ...
