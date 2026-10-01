use std::error::Error;
use std::fmt;

use numpy::ndarray::{Dim, Dimension};
use numpy::{
    Element, PyArray, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyErr;

#[derive(Debug)]
pub enum UnpackError {
    ElementType,
    Dimensions,
    TypeError,
    Borrowed,
}
impl fmt::Display for UnpackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnpackError::ElementType => write!(f, "Invalid element type"),
            UnpackError::Dimensions => write!(f, "Invalid dimensions"),
            UnpackError::TypeError => write!(f, "Invalid type"),
            UnpackError::Borrowed => write!(f, "Array is borrowed"),
        }
    }
}

impl From<UnpackError> for PyErr {
    fn from(err: UnpackError) -> PyErr {
        match err {
            UnpackError::TypeError | UnpackError::ElementType => {
                PyTypeError::new_err(format!("{}", err))
            }
            UnpackError::Dimensions | UnpackError::Borrowed => {
                PyValueError::new_err(format!("{}", err))
            }
        }
    }
}

impl Error for UnpackError {}

pub(crate) fn unpack_array<'py, T, const D: usize>(
    arr: &Bound<'py, PyAny>,
) -> Result<PyReadonlyArray<'py, T, Dim<[usize; D]>>, UnpackError>
where
    T: Element,
    Dim<[usize; D]>: Dimension,
{
    let untyped = arr
        .cast::<PyUntypedArray>()
        .map_err(|_| UnpackError::TypeError)?;
    if !untyped.dtype().is_equiv_to(&T::get_dtype(arr.py())) {
        return Err(UnpackError::ElementType);
    }
    if untyped.ndim() != D {
        return Err(UnpackError::Dimensions);
    }

    let typed = untyped
        .cast::<PyArray<T, Dim<[usize; D]>>>()
        .map_err(|_| UnpackError::ElementType)?;

    typed.try_readonly().map_err(|_| UnpackError::Borrowed)
}
