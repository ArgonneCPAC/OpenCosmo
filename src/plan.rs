use pyo3::prelude::*;
#[pymodule]
pub(crate) mod plan {
    use crate::arrays::unpack_array;
    use numpy::ndarray::{Array1, ArrayView1};
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    type PyWindows<'py> = (
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
    );

    /// Windows of consecutive storage chunks. Window `w` holds rows
    /// `[row_lo[w], row_hi[w])` and covers index positions `[pos_lo[w], pos_hi[w])`
    /// (elements of a simple index, segments of a chunked index). Its first selected
    /// row lands at `out_lo[w]` in the output.
    #[derive(Default)]
    struct Windows {
        row_lo: Vec<i64>,
        row_hi: Vec<i64>,
        pos_lo: Vec<i64>,
        pos_hi: Vec<i64>,
        out_lo: Vec<i64>,
    }

    /// The window currently being filled, in chunk ids.
    struct Open {
        chunk_lo: i64,
        chunk_last: i64,
        pos_lo: i64,
        out_lo: i64,
    }

    impl Windows {
        fn close(&mut self, open: &Open, pos_hi: i64, chunk_rows: i64, n_rows: i64) {
            self.row_lo.push(open.chunk_lo * chunk_rows);
            self.row_hi
                .push(((open.chunk_last + 1) * chunk_rows).min(n_rows));
            self.pos_lo.push(open.pos_lo);
            self.pos_hi.push(pos_hi);
            self.out_lo.push(open.out_lo);
        }

        fn into_py<'py>(self, py: Python<'py>) -> PyWindows<'py> {
            (
                Array1::from_vec(self.row_lo).into_pyarray(py),
                Array1::from_vec(self.row_hi).into_pyarray(py),
                Array1::from_vec(self.pos_lo).into_pyarray(py),
                Array1::from_vec(self.pos_hi).into_pyarray(py),
                Array1::from_vec(self.out_lo).into_pyarray(py),
            )
        }
    }

    /// Whether `chunk` can join `open` without skipping an untouched chunk or
    /// growing the window past `max_chunks`.
    fn extends(open: &Option<Open>, chunk: i64, max_chunks: i64) -> bool {
        open.as_ref()
            .is_some_and(|w| chunk <= w.chunk_last + 1 && chunk < w.chunk_lo + max_chunks)
    }

    fn validate_shape(n_rows: i64, chunk_rows: i64, max_chunks: i64) -> PyResult<()> {
        if n_rows < 0 || chunk_rows <= 0 || max_chunks <= 0 {
            return Err(PyValueError::new_err(
                "n_rows must be non-negative, and chunk_rows and max_chunks positive",
            ));
        }
        Ok(())
    }

    fn out_of_range(n_rows: i64) -> PyErr {
        PyValueError::new_err(format!(
            "Index is out of range for a column of {n_rows} rows"
        ))
    }

    #[pyfunction(name = "plan_windows_simple")]
    fn plan_windows_simple_py<'py>(
        py: Python<'py>,
        index: &Bound<'py, PyAny>,
        n_rows: i64,
        chunk_rows: i64,
        max_chunks: i64,
    ) -> PyResult<Option<PyWindows<'py>>> {
        let index = unpack_array::<i64, 1>(index)?;
        validate_shape(n_rows, chunk_rows, max_chunks)?;
        let windows = plan_windows_simple(index.as_array(), n_rows, chunk_rows, max_chunks)?;
        Ok(windows.map(|w| w.into_py(py)))
    }

    /// Returns `None` if `index` is not strictly increasing.
    fn plan_windows_simple(
        index: ArrayView1<'_, i64>,
        n_rows: i64,
        chunk_rows: i64,
        max_chunks: i64,
    ) -> PyResult<Option<Windows>> {
        let mut windows = Windows::default();
        let mut open: Option<Open> = None;
        let mut previous = -1;
        for (pos, &row) in index.iter().enumerate() {
            let pos = pos as i64;
            if row < 0 || row >= n_rows {
                return Err(out_of_range(n_rows));
            }
            if row <= previous {
                return Ok(None);
            }
            previous = row;

            let chunk = row / chunk_rows;
            if extends(&open, chunk, max_chunks) {
                if let Some(w) = open.as_mut() {
                    w.chunk_last = chunk;
                }
                continue;
            }
            if let Some(w) = &open {
                windows.close(w, pos, chunk_rows, n_rows);
            }
            open = Some(Open {
                chunk_lo: chunk,
                chunk_last: chunk,
                pos_lo: pos,
                out_lo: pos,
            });
        }
        if let Some(w) = &open {
            windows.close(w, index.len() as i64, chunk_rows, n_rows);
        }
        Ok(Some(windows))
    }

    #[pyfunction(name = "plan_windows_chunked")]
    fn plan_windows_chunked_py<'py>(
        py: Python<'py>,
        starts: &Bound<'py, PyAny>,
        sizes: &Bound<'py, PyAny>,
        n_rows: i64,
        chunk_rows: i64,
        max_chunks: i64,
    ) -> PyResult<Option<PyWindows<'py>>> {
        let starts = unpack_array::<i64, 1>(starts)?;
        let sizes = unpack_array::<i64, 1>(sizes)?;
        if starts.len()? != sizes.len()? {
            return Err(PyValueError::new_err(
                "start and size must be the same length in a chunked index!",
            ));
        }
        validate_shape(n_rows, chunk_rows, max_chunks)?;
        let windows = plan_windows_chunked(
            starts.as_array(),
            sizes.as_array(),
            n_rows,
            chunk_rows,
            max_chunks,
        )?;
        Ok(windows.map(|w| w.into_py(py)))
    }

    /// Returns `None` if the segments are not ordered and disjoint. A segment that
    /// crosses a window boundary belongs to both windows.
    fn plan_windows_chunked(
        starts: ArrayView1<'_, i64>,
        sizes: ArrayView1<'_, i64>,
        n_rows: i64,
        chunk_rows: i64,
        max_chunks: i64,
    ) -> PyResult<Option<Windows>> {
        let mut windows = Windows::default();
        let mut open: Option<Open> = None;
        let mut previous_end = 0;
        let mut out = 0;
        for (pos, (&start, &size)) in starts.iter().zip(sizes.iter()).enumerate() {
            let pos = pos as i64;
            if size < 0 {
                return Err(PyValueError::new_err("Chunk sizes must be non-negative"));
            }
            if size == 0 {
                continue;
            }
            let end = start + size;
            if start < 0 || end > n_rows {
                return Err(out_of_range(n_rows));
            }
            if start < previous_end {
                return Ok(None);
            }
            previous_end = end;

            let last_chunk = (end - 1) / chunk_rows;
            let mut row = start;
            while row < end {
                let chunk = row / chunk_rows;
                if !extends(&open, chunk, max_chunks) {
                    if let Some(w) = &open {
                        windows.close(w, pos, chunk_rows, n_rows);
                    }
                    open = Some(Open {
                        chunk_lo: chunk,
                        chunk_last: chunk,
                        pos_lo: pos,
                        out_lo: out + row - start,
                    });
                }
                let Some(w) = open.as_mut() else {
                    unreachable!("a window is always open here")
                };
                w.chunk_last = last_chunk.min(w.chunk_lo + max_chunks - 1);
                row = end.min((w.chunk_last + 1) * chunk_rows);
                if row < end {
                    // The window is full and the segment continues into the next one.
                    windows.close(w, pos + 1, chunk_rows, n_rows);
                    open = None;
                }
            }
            out += size;
        }
        if let Some(w) = &open {
            windows.close(w, starts.len() as i64, chunk_rows, n_rows);
        }
        Ok(Some(windows))
    }
}
