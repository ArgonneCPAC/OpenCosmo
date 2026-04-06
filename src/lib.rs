/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
///
///
use pyo3::prelude::*;
mod index;

#[pyo3::pymodule]
mod _lib {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::index::index;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
