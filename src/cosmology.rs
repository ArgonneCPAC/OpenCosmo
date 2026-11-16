use crate::arrays::unpack_array;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const C: f64 = 299_792_458.;
// Simpson panel width in ln(1 + z)
const LOG_STEP: f64 = 1.0e-3;
const MAX_REDSHIFT: f64 = 1.0e4;
const MAX_ITERATIONS: usize = 200;

#[pyfunction]
#[pyo3(signature = (h, Om0, Ode0 = None, Or0 = None, w0 = None, wa = None))]
#[allow(non_snake_case)]
fn make_cosmology(
    h: f64,
    Om0: f64,
    Ode0: Option<f64>,
    Or0: Option<f64>,
    w0: Option<f64>,
    wa: Option<f64>,
) -> PyResult<Cosmology> {
    if !(h.is_finite() && h > 0.) {
        return Err(PyValueError::new_err("h must be finite and positive"));
    }
    let Or0v = Or0.unwrap_or(0.);
    if !(Om0.is_finite() && Om0 >= 0. && Or0v.is_finite() && Or0v >= 0.) {
        return Err(PyValueError::new_err(
            "Om0 and Or0 must be finite and non-negative",
        ));
    }
    let Ode0v = Ode0.unwrap_or(1. - Om0 - Or0v);
    let (w0v, wav) = (w0.unwrap_or(-1.), wa.unwrap_or(0.));
    if !(Ode0v.is_finite() && w0v.is_finite() && wav.is_finite()) {
        return Err(PyValueError::new_err("Ode0, w0, and wa must be finite"));
    }

    let cosmology = Cosmology {
        h,
        Om0,
        Or0: Or0v,
        Ode0: Ode0v,
        Ok0: 1. - Om0 - Or0v - Ode0v,
        w0: w0v,
        wa: wav,
    };

    Ok(cosmology)
}

#[pymodule]
pub(crate) fn cosmology(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_cosmology, m)?)?;
    m.add_class::<Cosmology>()?;
    Ok(())
}

#[pyclass(frozen)]
#[allow(non_snake_case)]
struct Cosmology {
    h: f64,
    Om0: f64,
    Or0: f64,
    Ode0: f64,
    Ok0: f64,
    w0: f64,
    wa: f64,
}

impl Cosmology {
    fn hubble_distance(&self) -> f64 {
        // Mpc
        C / 100_000. / self.h
    }

    fn curvature_density(&self, z: f64) -> f64 {
        self.Ok0 * (1. + z).powi(2)
    }
    fn radiation_density(&self, z: f64) -> f64 {
        self.Or0 * (1. + z).powi(4)
    }
    fn matter_density(&self, z: f64) -> f64 {
        self.Om0 * (1. + z).powi(3)
    }
    fn dark_energy_density(&self, z: f64) -> f64 {
        match (self.w0, self.wa) {
            (-1., 0.) => self.Ode0,
            (w, 0.) => self.Ode0 * (1. + z).powf(3. * (1. + w)),
            (w0, wa) => {
                let coeff = (-3. * wa * (1. - (1. / (1. + z)))).exp();
                self.Ode0 * coeff * (1. + z).powf(3. * (1. + w0 + wa))
            }
        }
    }

    fn e_(&self, z: f64) -> f64 {
        let coeff = self.matter_density(z)
            + self.curvature_density(z)
            + self.dark_energy_density(z)
            + self.radiation_density(z);
        coeff.sqrt()
    }

    // (1 + z) / E(z), the comoving distance integrand with respect to ln(1 + z)
    fn log_integrand(&self, u: f64) -> f64 {
        let fac = u.exp();
        fac / self.e_(fac - 1.)
    }

    fn comoving_distance_unchecked(&self, z1: f64, z2: f64) -> f64 {
        if z2 <= z1 {
            return 0.;
        }
        let (u1, u2) = (z1.ln_1p(), z2.ln_1p());
        let panels = ((u2 - u1) / LOG_STEP).ceil().max(1.) as usize;
        let h = (u2 - u1) / panels as f64;

        let mut sum = self.log_integrand(u1) + self.log_integrand(u2);
        sum += 4. * self.log_integrand(u1 + 0.5 * h);
        for i in 1..panels {
            sum += 4. * self.log_integrand(u1 + (i as f64 + 0.5) * h);
            sum += 2. * self.log_integrand(u1 + i as f64 * h);
        }
        self.hubble_distance() * sum * h / 6.
    }

    fn comoving_distance_grid_unchecked(
        &self,
        z_min: f64,
        z_max: f64,
        n: usize,
    ) -> (Array1<f64>, Array1<f64>) {
        let z = Array1::linspace(z_min, z_max, n);
        let mut chi = Array1::zeros(n);
        chi[0] = self.comoving_distance_unchecked(0., z[0]);
        for i in 1..n {
            chi[i] = chi[i - 1] + self.comoving_distance_unchecked(z[i - 1], z[i]);
        }
        (z, chi)
    }

    // Safeguarded Newton iteration on chi(z) - distance, using dchi/dz = D_H / E(z).
    // Falls back to bisection whenever a Newton step leaves the bracket.
    fn redshift_at_distance_unchecked(&self, distance: f64) -> Result<f64, String> {
        if distance == 0. {
            return Ok(0.);
        }
        let hubble_distance = self.hubble_distance();

        let (mut lo, mut hi) = (0., 1.);
        while self.comoving_distance_unchecked(0., hi) < distance {
            lo = hi;
            hi *= 2.;
            if hi > MAX_REDSHIFT {
                return Err(format!(
                    "Comoving distance {distance} Mpc is not reached below z = {MAX_REDSHIFT}"
                ));
            }
        }

        let mut z = 0.5 * (lo + hi);
        for _ in 0..MAX_ITERATIONS {
            let residual = self.comoving_distance_unchecked(0., z) - distance;
            if !residual.is_finite() {
                return Err(format!("Comoving distance is not finite at z = {z}"));
            }
            if residual > 0. {
                hi = z;
            } else {
                lo = z;
            }

            let newton = z - residual * self.e_(z) / hubble_distance;
            let next = if newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            let tolerance = 1.0e-12 * (1. + next);
            if (next - z).abs() <= tolerance || hi - lo <= tolerance {
                return Ok(next);
            }
            z = next;
        }
        Err(format!(
            "Redshift solve for comoving distance {distance} Mpc did not converge"
        ))
    }
}

#[pymethods]
impl Cosmology {
    #[pyo3(signature = (z1, z2 = None))]
    fn comoving_distance(&self, z1: f64, z2: Option<f64>) -> PyResult<f64> {
        let (lower, upper) = match z2 {
            None => (0., z1),
            Some(z2v) => (z1, z2v),
        };
        if !(lower.is_finite() && upper.is_finite()) || lower < 0. || upper < lower {
            return Err(PyValueError::new_err(
                "Redshifts must be finite and non-negative, with z1 <= z2",
            ));
        }
        finite_distance(self.comoving_distance_unchecked(lower, upper))
    }
    fn comoving_distances<'py>(
        &self,
        py: Python<'py>,
        redshifts: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let arr = unpack_array::<f64, 1>(redshifts)?;
        let mut output: Array1<f64> = Array1::zeros(arr.len()?);
        for (i, &val) in arr.as_array().iter().enumerate() {
            output[i] = self.comoving_distance(val, None)?;
        }

        Ok(output.into_pyarray(py))
    }
    fn comoving_distance_grid<'py>(
        &self,
        py: Python<'py>,
        z_min: f64,
        z_max: f64,
        n: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        if !(z_min.is_finite() && z_max.is_finite()) || z_min < 0. || z_max < z_min {
            return Err(PyValueError::new_err(
                "Redshifts must be finite and non-negative, with z_min <= z_max",
            ));
        }
        if n < 2 {
            return Err(PyValueError::new_err("Grid must have at least two points"));
        }
        let (z, chi) = self.comoving_distance_grid_unchecked(z_min, z_max, n);
        // The grid is cumulative, so a non-finite value anywhere reaches the end
        finite_distance(chi[n - 1])?;
        Ok((z.into_pyarray(py), chi.into_pyarray(py)))
    }
    fn redshift_at_distance(&self, distance: f64) -> PyResult<f64> {
        if !distance.is_finite() || distance < 0. {
            return Err(PyValueError::new_err(
                "Distance must be finite and non-negative",
            ));
        }
        self.redshift_at_distance_unchecked(distance)
            .map_err(PyValueError::new_err)
    }
}

// A non-finite distance means E(z)^2 <= 0 somewhere in the integration range
fn finite_distance(distance: f64) -> PyResult<f64> {
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(PyValueError::new_err(
            "Comoving distance is not finite; E(z)^2 is non-positive in this range",
        ))
    }
}
