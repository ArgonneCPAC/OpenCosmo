use std::any::type_name;
use std::error::Error;
use std::fmt;

use numpy::ndarray::{Dim, Dimension};
use numpy::{
    Element, PyArray, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::PyErr;

#[derive(Debug)]
pub enum UnpackError {
    ElementType((String, String)),
    Dimensions((usize, usize)),
    TypeError((String, String)),
    Borrowed,
}
impl fmt::Display for UnpackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnpackError::ElementType((exp, fnd)) => {
                write!(f, "Invalid element type. Expected {exp} and got {fnd}")
            }
            UnpackError::Dimensions((exp, fnd)) => {
                write!(f, "Expected a {exp} dimensional array but got {fnd}")
            }
            UnpackError::TypeError((exp, fnd)) => {
                write!(f, "Invalid type. Expected {exp} got {fnd}")
            }
            UnpackError::Borrowed => write!(f, "Array is borrowed"),
        }
    }
}

impl From<UnpackError> for PyErr {
    fn from(err: UnpackError) -> PyErr {
        match err {
            UnpackError::TypeError(_) | UnpackError::ElementType(_) => {
                PyTypeError::new_err(format!("{}", err))
            }
            UnpackError::Dimensions(_) | UnpackError::Borrowed => {
                PyValueError::new_err(format!("{}", err))
            }
        }
    }
}

impl Error for UnpackError {}

fn get_type_string<'py>(pytype: Bound<'py, PyType>) -> String {
    pytype
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown type>".to_owned())
}

pub(crate) fn unpack_array<'py, T, const D: usize>(
    arr: &Bound<'py, PyAny>,
) -> Result<PyReadonlyArray<'py, T, Dim<[usize; D]>>, UnpackError>
where
    T: Element,
    Dim<[usize; D]>: Dimension,
{
    let untyped = arr.cast::<PyUntypedArray>().map_err(|_| {
        UnpackError::TypeError(("numpy array".to_owned(), get_type_string(arr.get_type())))
    })?;
    let dtype = untyped.dtype();
    if !dtype.is_equiv_to(&T::get_dtype(arr.py())) {
        return Err(UnpackError::ElementType((
            type_name::<T>().to_owned(),
            format!("{}", dtype),
        )));
    }
    if untyped.ndim() != D {
        return Err(UnpackError::Dimensions((D, untyped.ndim())));
    }

    let typed = untyped.cast::<PyArray<T, Dim<[usize; D]>>>().map_err(|_| {
        UnpackError::TypeError((
            format!("{}-dimensional numpy array", D),
            get_type_string(arr.get_type()),
        ))
    })?;

    typed.try_readonly().map_err(|_| UnpackError::Borrowed)
}
