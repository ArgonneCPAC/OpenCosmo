use pyo3::prelude::*;
#[pymodule]
pub(crate) mod index {
    use crate::arrays::unpack_array;
    use numpy::ndarray::s;
    use numpy::ndarray::{Array1, ArrayView1};
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyList;
    use std::collections::HashMap;
    use std::iter::zip;

    type PyIndexPair<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

    fn checked_range_end(start: i64, size: i64) -> PyResult<i64> {
        if start < 0 || size < 0 {
            return Err(PyValueError::new_err(
                "Index range starts and sizes must be nonnegative",
            ));
        }
        start
            .checked_add(size)
            .ok_or_else(|| PyValueError::new_err("Index range end overflowed int64"))
    }

    fn checked_allocation_length(sizes: ArrayView1<'_, i64>) -> PyResult<usize> {
        sizes.iter().try_fold(0usize, |total, &size| {
            let size = usize::try_from(size)
                .map_err(|_| PyValueError::new_err("Index range sizes must be nonnegative"))?;
            total
                .checked_add(size)
                .ok_or_else(|| PyValueError::new_err("Index allocation length overflowed usize"))
        })
    }

    fn unpack_index_array<'py>(index: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray1<'py, i64>> {
        Ok(unpack_array::<i64, 1>(index)?)
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
        if index.is_empty() {
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
        if start.is_empty() {
            return Ok((0, 0));
        }
        let mut index_range = (start[0], checked_range_end(start[0], size[0])?);
        for (&st, &si) in zip(start, size) {
            let end = checked_range_end(st, si)?;
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
        )?;
        Ok(result.into_pyarray(py))
    }

    fn n_in_range_chunked(
        start: ArrayView1<'_, i64>,
        size: ArrayView1<'_, i64>,
        range_start: ArrayView1<'_, i64>,
        range_size: ArrayView1<'_, i64>,
    ) -> PyResult<Array1<i64>> {
        let mut output = Array1::<i64>::zeros(range_start.len());
        if start.is_empty() {
            return Ok(output);
        }
        for (i, (&rst, &rsi)) in zip(range_start, range_size).enumerate() {
            let range_end = checked_range_end(rst, rsi)?;
            let mut total = 0i64;
            for (&st, &si) in zip(start, size) {
                let end = checked_range_end(st, si)?;
                let overlap = end.min(range_end) - st.max(rst);
                if overlap > 0 {
                    total = total.checked_add(overlap).ok_or_else(|| {
                        PyValueError::new_err("Index overlap count overflowed int64")
                    })?;
                }
            }
            output[i] = total;
        }
        Ok(output)
    }
    #[pyfunction(name = "chunked_into_array")]
    pub(crate) fn chunked_into_array_py<'py>(
        py: Python<'py>,
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        let output = chunked_into_array(start_arr.as_array(), size_arr.as_array())?;
        Ok(output.into_pyarray(py))
    }
    fn chunked_into_array(
        start: ArrayView1<'_, i64>,
        size: ArrayView1<'_, i64>,
    ) -> PyResult<Array1<i64>> {
        let total_length = checked_allocation_length(size)?;
        let mut output = Array1::<i64>::zeros(total_length);
        let mut output_index = 0usize;
        for (&st, &si) in zip(start, size) {
            let end = checked_range_end(st, si)?;
            for value in st..end {
                output[output_index] = value;
                output_index += 1;
            }
        }
        Ok(output)
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
        let total_length = checked_allocation_length(size)?;
        let mut output = Array1::<i64>::zeros(total_length);
        let mut output_index = 0usize;
        for (&st, &si) in zip(start, size) {
            let end = checked_range_end(st, si)?;
            let start_index = usize::try_from(st)
                .map_err(|_| PyValueError::new_err("Index range starts must be nonnegative"))?;
            let end_index = usize::try_from(end)
                .map_err(|_| PyValueError::new_err("Index range end exceeds usize"))?;
            if end_index > simple.len() {
                return Err(PyValueError::new_err(
                    "The chunked index is outside of the range of the simple index!",
                ));
            }
            let to_insert = simple.slice(s![start_index..end_index]);
            let output_end = output_index
                .checked_add(to_insert.len())
                .ok_or_else(|| PyValueError::new_err("Index output position overflowed usize"))?;
            output
                .slice_mut(s![output_index..output_end])
                .assign(&to_insert);
            output_index = output_end;
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
    ) -> PyResult<PyIndexPair<'py>> {
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
    fn find_chunk(prefix: &[i64], x: i64) -> usize {
        prefix.partition_point(|&offset| offset <= x) - 1
    }

    fn take_chunked_from_chunked(
        start: ArrayView1<'_, i64>,
        size: ArrayView1<'_, i64>,
        take_start: ArrayView1<'_, i64>,
        take_size: ArrayView1<'_, i64>,
    ) -> Result<(Array1<i64>, Array1<i64>), PyErr> {
        if take_start.is_empty() {
            return Ok((Array1::<i64>::zeros(0), Array1::<i64>::zeros(0)));
        }
        let mut output_start: Vec<i64> = Vec::new();
        let mut output_size: Vec<i64> = Vec::new();

        let mut prefix = vec![0i64; size.len() + 1];
        for i in 0..size.len() {
            checked_range_end(start[i], size[i])?;
            prefix[i + 1] = prefix[i]
                .checked_add(size[i])
                .ok_or_else(|| PyValueError::new_err("Index length overflowed int64"))?;
        }
        let total = prefix[size.len()];

        for (&tstart, &tsize) in zip(take_start, take_size) {
            let take_end = checked_range_end(tstart, tsize)?;
            if take_end > total {
                return Err(PyValueError::new_err(
                    "You can't take more elements than exist in an index!",
                ));
            }
            if tsize == 0 {
                continue;
            }
            if size.is_empty() {
                return Err(PyValueError::new_err(
                    "You can't take more elements than exist in an index!",
                ));
            }
            let mut chunk_index = find_chunk(&prefix, tstart);
            let cs = prefix[chunk_index];
            let mut start_in_chunk = tstart - cs;
            let mut chunk_taken = 0i64;

            loop {
                let size_in_chunk = size[chunk_index] - start_in_chunk;
                let remaining = tsize - chunk_taken;
                let (take, chunk_completed) = if size_in_chunk >= remaining {
                    (remaining, true)
                } else {
                    (size_in_chunk, false)
                };

                output_start.push(
                    start[chunk_index]
                        .checked_add(start_in_chunk)
                        .ok_or_else(|| PyValueError::new_err("Index range end overflowed int64"))?,
                );
                output_size.push(take);
                chunk_taken += take;

                if chunk_completed {
                    break;
                }
                chunk_index += 1;
                start_in_chunk = 0;
            }
        }

        Ok((
            Array1::from_vec(output_start),
            Array1::from_vec(output_size),
        ))
    }
    #[pyfunction(name = "reindex_column")]
    fn reindex_columns_py<'py>(
        py: Python<'py>,
        index: &Bound<'_, PyAny>,
        index_column: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let index_arr = unpack_index_array(index)?;
        let index_column_arr = unpack_index_array(index_column)?;
        Ok(reindex_column(index_arr.as_array(), index_column_arr.as_array()).into_pyarray(py))
    }

    fn reindex_column(
        index: ArrayView1<'_, i64>,
        index_column: ArrayView1<'_, i64>,
    ) -> Array1<i64> {
        let mut index_map: HashMap<i64, i64> = HashMap::new();
        for (i, &index_entry) in index.iter().enumerate() {
            index_map.insert(index_entry, i as i64);
        }

        let mut output: Vec<i64> = Vec::with_capacity(index_column.len());
        for val in index_column.iter() {
            let val_index_opt = index_map.get(val);
            if let Some(&val_index) = val_index_opt {
                output.push(val_index)
            } else {
                output.push(-1)
            }
        }
        Array1::from_vec(output)
    }

    #[pyfunction(name = "rebuild_chunked_by_ranges")]
    fn rebuild_chunked_by_ranges_py<'py>(
        py: Python<'py>,
        starts: &Bound<'_, PyAny>,
        sizes: &Bound<'_, PyAny>,
        range_starts: &Bound<'_, PyAny>,
        range_sizes: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let (start_arr, size_arr) = unpack_chunked_index(starts, sizes)?;
        let (range_starts_arr, range_sizes_arr) = unpack_chunked_index(range_starts, range_sizes)?;
        let mut output = rebuild_chunked_by_ranges(
            start_arr.as_array(),
            size_arr.as_array(),
            range_starts_arr.as_array(),
            range_sizes_arr.as_array(),
        )?;
        PyList::new(
            py,
            output
                .drain(0..)
                .map(|(st, si)| (st.into_pyarray(py), si.into_pyarray(py))),
        )
    }

    fn rebuild_chunked_by_ranges(
        starts: ArrayView1<'_, i64>,
        sizes: ArrayView1<'_, i64>,
        range_starts: ArrayView1<'_, i64>,
        range_sizes: ArrayView1<'_, i64>,
    ) -> PyResult<Vec<(Array1<i64>, Array1<i64>)>> {
        let n_datasets = range_starts.len();
        let mut outputs: Vec<(Vec<i64>, Vec<i64>)> =
            (0..n_datasets).map(|_| (Vec::new(), Vec::new())).collect();

        if starts.is_empty() || n_datasets == 0 {
            return Ok(outputs
                .into_iter()
                .map(|(st, si)| (Array1::from_vec(st), Array1::from_vec(si)))
                .collect());
        }

        let mut i = 0usize;
        let mut j = 0usize;

        while i < starts.len() && j < n_datasets {
            let chunk_start = starts[i];
            let chunk_end = checked_range_end(chunk_start, sizes[i])?;
            let ds_start = range_starts[j];
            let ds_end = checked_range_end(ds_start, range_sizes[j])?;

            if chunk_end <= ds_start {
                i += 1;
            } else if chunk_start >= ds_end {
                j += 1;
            } else {
                let overlap_start = chunk_start.max(ds_start);
                let overlap_end = chunk_end.min(ds_end);
                outputs[j].0.push(overlap_start - ds_start);
                outputs[j].1.push(overlap_end - overlap_start);

                if chunk_end <= ds_end {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }

        Ok(outputs
            .into_iter()
            .map(|(st, si)| (Array1::from_vec(st), Array1::from_vec(si)))
            .collect())
    }
    #[pyfunction(name = "rebuild_simple_by_ranges")]
    fn rebuild_simple_by_ranges_py<'py>(
        py: Python<'py>,
        index: &Bound<'_, PyAny>,
        range_starts: &Bound<'_, PyAny>,
        range_sizes: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let index_arr = unpack_index_array(index)?;
        let (start_arr, size_arr) = unpack_chunked_index(range_starts, range_sizes)?;
        let mut output = rebuild_simple_by_ranges(
            index_arr.as_array(),
            start_arr.as_array(),
            size_arr.as_array(),
        )?;
        PyList::new(py, output.drain(0..).map(|a| a.into_pyarray(py)))
    }

    #[pyfunction(name = "project_chunked_on_simple")]
    fn project_chunked_on_simple_py<'py>(
        py: Python<'py>,
        simple: &Bound<'_, PyAny>,
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let simple_arr = unpack_index_array(simple)?;
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        let result = project_chunked_on_simple(
            simple_arr.as_array(),
            start_arr.as_array(),
            size_arr.as_array(),
        )?;
        Ok(result.into_pyarray(py))
    }

    fn project_chunked_on_simple(
        simple: ArrayView1<'_, i64>,
        start: ArrayView1<'_, i64>,
        size: ArrayView1<'_, i64>,
    ) -> PyResult<Array1<i64>> {
        let mut output: Vec<i64> = Vec::new();
        let n_chunks = start.len();
        if simple.is_empty() || n_chunks == 0 {
            return Ok(Array1::from_vec(output));
        }

        let mut chunk_idx = 0usize;
        for (i, &val) in simple.iter().enumerate() {
            // Advance past chunks whose end is at or before val.
            while chunk_idx < n_chunks
                && val >= checked_range_end(start[chunk_idx], size[chunk_idx])?
            {
                chunk_idx += 1;
            }
            if chunk_idx >= n_chunks {
                break;
            }
            // val is before the current chunk; skip.
            if val < start[chunk_idx] {
                continue;
            }
            output.push(i as i64);
        }
        Ok(Array1::from_vec(output))
    }

    fn rebuild_simple_by_ranges(
        index: ArrayView1<'_, i64>,
        range_starts: ArrayView1<'_, i64>,
        range_sizes: ArrayView1<'_, i64>,
    ) -> PyResult<Vec<Array1<i64>>> {
        let n_ranges = range_starts.len();
        let mut outputs: Vec<Vec<i64>> = (0..n_ranges).map(|_| Vec::new()).collect();

        if n_ranges == 0 || index.is_empty() {
            return Ok(outputs.into_iter().map(Array1::from_vec).collect());
        }

        let mut j = 0usize;
        for &idx in index {
            // Advance past all ranges whose end is at or before idx.
            // Using a while loop handles the case where idx skips multiple ranges
            // and prevents an OOB panic when advancing past the last range.
            while j < n_ranges && idx >= checked_range_end(range_starts[j], range_sizes[j])? {
                j += 1;
            }
            if j >= n_ranges {
                break;
            }
            // idx falls before the start of range j (gap in a non-contiguous layout).
            if idx < range_starts[j] {
                continue;
            }
            outputs[j].push(idx - range_starts[j]);
        }

        Ok(outputs.into_iter().map(Array1::from_vec).collect())
    }
}
