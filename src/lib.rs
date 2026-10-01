/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
///
///
use pyo3::prelude::*;

mod index;
mod spatial;

#[pymodule]
mod _lib {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::index::index;

    #[pymodule_export]
    use crate::spatial::spatial;
}
