use pyo3::prelude::*;
#[pymodule]
pub(crate) mod index {
    use numpy::ndarray::s;
    use numpy::ndarray::{Array1, ArrayView1};
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::prelude::*;
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

    #[pyfunction(name = "get_chunked_range")]
    pub(crate) fn get_chunked_range_py(
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
    ) -> PyResult<(i64, i64)> {
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        get_chunked_range(start_arr.as_array(), size_arr.as_array())
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
    #[pyfunction(name = "chunked_into_array")]
    pub(crate) fn chunked_into_array_py<'py>(
        py: Python<'py>,
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        let output = chunked_into_array(start_arr.as_array(), size_arr.as_array());
        Ok(output.into_pyarray(py))
    }
    fn chunked_into_array(start: ArrayView1<'_, i64>, size: ArrayView1<'_, i64>) -> Array1<i64> {
        let total_length = size.sum();
        let mut output = Array1::<i64>::zeros(total_length as usize);
        let mut rs: i64 = 0;
        for (&st, &si) in zip(start, size) {
            let range = Array1::from_iter(st..st + si);
            output
                .slice_mut(s![rs as usize..(rs + si) as usize])
                .assign(&range);
            rs += si;
        }
        output
    }

    #[pyfunction(name = "take_chunked_from_simple")]
    fn take_chunked_from_simple_py<'py>(
        py: Python<'py>,
        simple: &Bound<'_, PyAny>,
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let simple_arr = unpack_index_array(simple)?;
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        let result = take_chunked_from_simple(
            simple_arr.as_array(),
            start_arr.as_array(),
            size_arr.as_array(),
        );
        Ok(result?.into_pyarray(py))
    }

    fn take_chunked_from_simple(
        simple: ArrayView1<'_, i64>,
        start: ArrayView1<'_, i64>,
        size: ArrayView1<'_, i64>,
    ) -> Result<Array1<i64>, PyErr> {
        let total_length = size.sum();
        let mut output = Array1::<i64>::zeros(total_length as usize);
        let mut rs: i64 = 0;
        for (&st, &si) in zip(start, size) {
            let end = st + si;
            if end as usize > simple.len() {
                return Err(PyValueError::new_err(
                    "The chunked index is outside of the range of the simple index!",
                ));
            }
            let to_insert = simple.slice(s![st as usize..end as usize]);
            output
                .slice_mut(s![rs as usize..(rs + si) as usize])
                .assign(&to_insert);
            rs += si
        }
        Ok(output)
    }

    #[pyfunction(name = "take_chunked_from_chunked")]
    fn take_chunked_from_chunked_py<'py>(
        py: Python<'py>,
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
        take_start: &Bound<'_, PyAny>,
        take_size: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        let (take_start_arr, take_size_arr) = unpack_chunked_index(take_start, take_size)?;
        let result = take_chunked_from_chunked(
            start_arr.as_array(),
            size_arr.as_array(),
            take_start_arr.as_array(),
            take_size_arr.as_array(),
        )?;
        Ok((result.0.into_pyarray(py), result.1.into_pyarray(py)))
    }
    fn take_chunked_from_chunked(
        start: ArrayView1<'_, i64>,
        size: ArrayView1<'_, i64>,
        take_start: ArrayView1<'_, i64>,
        take_size: ArrayView1<'_, i64>,
    ) -> Result<(Array1<i64>, Array1<i64>), PyErr> {
        // assumption: everything is sorted
        let mut output_start: Vec<i64> = Vec::new();
        let mut output_size: Vec<i64> = Vec::new();
        if size.sum() < take_start[take_start.len() - 1] + take_size[take_size.len() - 1] {
            return Err(PyValueError::new_err(
                "You can't take more elements than exist in an index!",
            ));
        }
        let mut chunk_index: usize = 0;
        let mut cs = 0;
        for (&tstart, &tsize) in zip(take_start, take_size) {
            while cs + size[chunk_index] < tstart {
                cs += size[chunk_index];
                chunk_index += 1;
            }
            // chunk_index now points to the chunk that we need to start at
            // cs is equal to the cumulative size of all chunks that we've passed
            let mut start_in_chunk = tstart - cs;
            let mut chunk_taken = 0;
            let mut chunk_completed = false;
            while !chunk_completed {
                let mut size_in_chunk = size[chunk_index] - start_in_chunk;
                chunk_completed = size_in_chunk >= (tsize - chunk_taken);
                if chunk_completed {
                    size_in_chunk = tsize - chunk_taken;
                }

                output_start.push(start[chunk_index] + start_in_chunk);
                output_size.push(size_in_chunk);
                chunk_taken += size_in_chunk;
                if !chunk_completed {
                    cs += size[chunk_index];
                    chunk_index += 1;
                    start_in_chunk = 0;
                }
            }
        }
        let output_start_arr = Array1::from_vec(output_start);
        let output_size_arr = Array1::from_vec(output_size);
        Ok((output_start_arr, output_size_arr))
    }
}
