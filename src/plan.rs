use pyo3::prelude::*;

/// Read plans: ascending HDF5 reads for an index, with bounded temporary buffers.
#[pymodule]
pub(crate) mod plan {
    use crate::index::index::{unpack_chunked_index, unpack_index_array};
    use numpy::ndarray::{Array1, ArrayView1};
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use std::iter::zip;

    type PyReadPlan<'py> = (
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
    );

    /// Ascending source reads, each served through a bounded buffer.
    ///
    /// Run `k` reads rows `run_starts[k]..run_starts[k] + run_sizes[k]` and owns the
    /// next `run_segments[k]` segments. A segment copies `segment_sizes` rows from
    /// `segment_local` in the run buffer to `segment_output` in the output array.
    #[derive(Default)]
    struct ReadPlan {
        run_starts: Vec<i64>,
        run_sizes: Vec<i64>,
        run_segments: Vec<i64>,
        segment_output: Vec<i64>,
        segment_local: Vec<i64>,
        segment_sizes: Vec<i64>,
    }

    impl ReadPlan {
        fn with_capacity(n_segments: usize) -> Self {
            ReadPlan {
                segment_output: Vec::with_capacity(n_segments),
                segment_local: Vec::with_capacity(n_segments),
                segment_sizes: Vec::with_capacity(n_segments),
                ..Default::default()
            }
        }

        fn into_py(self, py: Python<'_>) -> PyReadPlan<'_> {
            // numpy adopts the Vec allocation, so drop growth headroom first
            let to_py = |mut vec: Vec<i64>| {
                vec.shrink_to_fit();
                Array1::from_vec(vec).into_pyarray(py)
            };
            (
                to_py(self.run_starts),
                to_py(self.run_sizes),
                to_py(self.run_segments),
                to_py(self.segment_output),
                to_py(self.segment_local),
                to_py(self.segment_sizes),
            )
        }
    }

    #[pyfunction(name = "get_chunked_read_plan")]
    #[pyo3(signature = (start, size, hdf5_chunk_size = None, max_buffer_rows = None))]
    fn get_chunked_read_plan_py<'py>(
        py: Python<'py>,
        start: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
        hdf5_chunk_size: Option<usize>,
        max_buffer_rows: Option<usize>,
    ) -> PyResult<PyReadPlan<'py>> {
        let (start_arr, size_arr) = unpack_chunked_index(start, size)?;
        let sizes = size_arr.as_array();
        if sizes.iter().any(|&val| val < 0) {
            return Err(PyValueError::new_err(
                "chunk sizes in a chunked index must be non-negative",
            ));
        }
        let mut offsets = Array1::<i64>::zeros(sizes.len());
        let mut total: i64 = 0;
        for (offset, &size) in zip(offsets.iter_mut(), sizes.iter()) {
            *offset = total;
            total += size;
        }
        Ok(get_read_plan(
            start_arr.as_array(),
            |i| sizes[i],
            |i| offsets[i],
            hdf5_chunk_size,
            max_buffer_rows,
        )
        .into_py(py))
    }

    #[pyfunction(name = "get_simple_read_plan")]
    #[pyo3(signature = (index, hdf5_chunk_size = None, max_buffer_rows = None))]
    fn get_simple_read_plan_py<'py>(
        py: Python<'py>,
        index: &Bound<'_, PyAny>,
        hdf5_chunk_size: Option<usize>,
        max_buffer_rows: Option<usize>,
    ) -> PyResult<PyReadPlan<'py>> {
        let index_arr = unpack_index_array(index)?;
        Ok(get_read_plan(
            index_arr.as_array(),
            |_| 1,
            |i| i as i64,
            hdf5_chunk_size,
            max_buffer_rows,
        )
        .into_py(py))
    }

    /// Plan reads of possibly unordered, overlapping row ranges.
    ///
    /// Ranges are visited in ascending start order and packed whole into runs. A
    /// range joins the current run unless it starts at least `hdf5_chunk_size` rows
    /// past it, or joining would make the run read or serve more than
    /// `max_buffer_rows` rows. A range larger than the limit gets a run of its own,
    /// which the caller reads straight into the output, so every buffered run
    /// holds at most `max_buffer_rows` rows.
    fn get_read_plan(
        starts: ArrayView1<'_, i64>,
        size_of: impl Fn(usize) -> i64,
        output_offset_of: impl Fn(usize) -> i64,
        hdf5_chunk_size: Option<usize>,
        max_buffer_rows: Option<usize>,
    ) -> ReadPlan {
        let n = starts.len();
        // Most indices arrive sorted, so only allocate a sort order when needed
        let order: Option<Vec<usize>> = if (1..n).all(|k| starts[k - 1] <= starts[k]) {
            None
        } else {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&i| starts[i]);
            Some(order)
        };

        let gap = hdf5_chunk_size.map(|val| val as i64);
        let max_rows = max_buffer_rows.map_or(i64::MAX, |val| val.max(1) as i64);
        let mut plan = ReadPlan::with_capacity(n);
        let mut run_start: i64 = 0;
        let mut run_end: i64 = 0;
        let mut run_served: i64 = 0;

        for k in 0..n {
            let i = order.as_ref().map_or(k, |order| order[k]);
            let size = size_of(i);
            if size == 0 {
                continue;
            }
            let (start, end) = (starts[i], starts[i] + size);
            if plan.run_starts.is_empty()
                || gap.is_some_and(|val| start - run_end >= val)
                || run_end.max(end) - run_start > max_rows
                || run_served + size > max_rows
            {
                plan.run_starts.push(start);
                plan.run_sizes.push(0);
                plan.run_segments.push(0);
                run_start = start;
                run_end = start;
                run_served = 0;
            }
            let output = output_offset_of(i);
            let local = start - run_start;

            // Coalesce with the previous segment when both sides stay contiguous
            let run_segments = plan.run_segments.last_mut().unwrap();
            let last = plan.segment_sizes.len().wrapping_sub(1);
            if *run_segments > 0
                && plan.segment_output[last] + plan.segment_sizes[last] == output
                && plan.segment_local[last] + plan.segment_sizes[last] == local
            {
                plan.segment_sizes[last] += size;
            } else {
                *run_segments += 1;
                plan.segment_output.push(output);
                plan.segment_local.push(local);
                plan.segment_sizes.push(size);
            }

            // Nested ranges never shrink the run
            run_end = run_end.max(end);
            run_served += size;
            *plan.run_sizes.last_mut().unwrap() = run_end - run_start;
        }
        plan
    }
}
