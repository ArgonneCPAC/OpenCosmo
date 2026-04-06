use pyo3::prelude::*;
#[pymodule]
pub(crate) mod index {
    use numpy::ndarray::{Array1, ArrayView1};
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::prelude::*;
    use std::io::Result;
    use std::iter::zip;
    fn unpack_index_array<'py>(index: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray1<'py, i64>> {
        let index_data = index
            .cast::<PyArray1<i64>>()
            .map_err(|_| PyTypeError::new_err("Indices should be a 1-d array of i64"))?;

        Ok(index_data.readonly())
    }

    fn unpack_chunked_index<'py>(
        start: &Bound<'py, PyAny>,
        size: &Bound<'py, PyAny>,
    ) -> PyResult<(PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>)> {
        let start_arr = unpack_index_array(start)?;
        let size_arr = unpack_index_array(size)?;
        if start_arr.len()? != size_arr.len()? {
            return Err(PyValueError::new_err(
                "start and size must be the same length in a chunked index!",
            ));
        }
        Ok((start_arr, size_arr))
    }

    #[pyfunction(name = "get_simple_range")]
    pub(crate) fn get_simple_range_py(index: &Bound<'_, PyAny>) -> PyResult<(i64, i64)> {
        let index_arr = unpack_index_array(index)?;
        get_simple_range(index_arr.as_array())
    }
    #[pyfunction(name = "get_chunked_range")]
    pub(crate) fn get_chunked_range_py(
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
    ) -> PyResult<(i64, i64)> {
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        get_chunked_range(start_arr.as_array(), size_arr.as_array())
    }
    #[pyfunction(name = "n_in_range_chunked")]
    pub(crate) fn n_in_range_chunked_py<'py>(
        py: Python<'py>,
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
        range_start: &Bound<'_, PyAny>,
        range_size: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        let (range_start_arr, range_size_arr) = unpack_chunked_index(range_start, range_size)?;
        let result = n_in_range_chunked(
            start_arr.as_array(),
            size_arr.as_array(),
            range_start_arr.as_array(),
            range_size_arr.as_array(),
        );
        Ok(result.into_pyarray(py))
    }

    fn n_in_range_chunked(
        start: ArrayView1<'_, i64>,
        size: ArrayView1<'_, i64>,
        range_start: ArrayView1<'_, i64>,
        range_size: ArrayView1<'_, i64>,
    ) -> Array1<i64> {
        let mut output = Array1::<i64>::zeros(range_start.len());
        if start.len() == 0 {
            return output;
        }
        let end = &start + &size;
        for (i, (&rst, &rsi)) in zip(range_start, range_size).enumerate() {
            let chunk_end = rst + rsi;
            let total = zip(start, &end)
                .filter(|(s, e)| !(**s > chunk_end || **e < rst))
                .map(|(&s, &e)| {
                    let mut cr = (s, e);
                    if chunk_end < e {
                        cr = (cr.0, chunk_end)
                    }
                    if rst > s {
                        cr = (rst, cr.1)
                    }
                    cr.1 - cr.0
                })
                .sum();
            output[i] = total;
        }
        output
    }
    fn get_simple_range(index: ArrayView1<'_, i64>) -> PyResult<(i64, i64)> {
        if index.len() == 0 {
            return Ok((0, 0));
        }
        let mut index_range = (index[0], index[0]);
        for &item in index {
            if item < index_range.0 {
                index_range = (item, index_range.1)
            }
            if item > index_range.1 {
                index_range = (index_range.0, item)
            }
        }
        Ok(index_range)
    }

    fn get_chunked_range(
        start: ArrayView1<'_, i64>,
        size: ArrayView1<'_, i64>,
    ) -> PyResult<(i64, i64)> {
        if start.len() == 0 {
            return Ok((0, 0));
        }
        let mut index_range = (start[0], start[0] + size[0]);
        for (&st, &si) in zip(start, size) {
            let end = st + si;
            if st < index_range.0 {
                index_range = (st, index_range.1);
            }
            if end > index_range.1 {
                index_range = (index_range.0, end);
            }
        }
        Ok(index_range)
    }
}
