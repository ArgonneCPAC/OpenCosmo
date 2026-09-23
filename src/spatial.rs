use pyo3::prelude::*;

#[pymodule]
pub(crate) mod spatial {
    use core::f64;

    use kiddo::dist::SquaredEuclidean;
    use kiddo::MutableKdTree;
    use numpy::ndarray::{Array1, ArrayView2};
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
    };
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::prelude::*;
    struct SafeRawBuffer(*mut f64);

    impl SafeRawBuffer {
        fn get(&self) -> *mut f64 {
            self.0
        }
    }

    // SAFETY: You must ensure that moving the pointer to another thread
    // does not cause data races or memory invalidation.
    unsafe impl Send for SafeRawBuffer {}

    // SAFETY: You must ensure that accessing the pointer concurrently via
    // shared references (&SafeRawBuffer) is synchronized properly.
    unsafe impl Sync for SafeRawBuffer {}

    fn unpack_points<'py>(array: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray2<'py, f64>> {
        let data = array
            .cast::<PyArray2<f64>>()
            .map_err(|err| PyTypeError::new_err(format!("Expected a 2d array {:?}", err)))?;

        if data.shape()[1] != 3 {
            return Err(PyValueError::new_err("Expected an array of points!"));
        }

        Ok(data.readonly())
    }

    #[pyfunction(name = "get_closest_distance_3d")]
    pub(crate) fn get_closest_distance_3d<'py>(
        py: Python<'py>,
        check_vecs: &Bound<'_, PyAny>,
        query_vecs: &Bound<'_, PyAny>,
        max_distance: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let check_arr = unpack_points(check_vecs)?;
        let query_arr = unpack_points(query_vecs)?;
        let arr = query_closest_3d(check_arr.as_array(), query_arr.as_array(), max_distance)?;
        Ok(arr.into_pyarray(py))
    }
    fn query_closest_3d(
        check_vecs: ArrayView2<'_, f64>,
        query_vecs: ArrayView2<'_, f64>,
        max_distance: f64,
    ) -> PyResult<Array1<f64>> {
        let mut tree: MutableKdTree<f64, 3> = MutableKdTree::default();
        for (i, row) in check_vecs.rows().into_iter().enumerate() {
            let p = [row[0], row[1], row[2]];
            tree.add(&p, i as u32)
                .map_err(|e| PyValueError::new_err(format!("Unable to add point: {:?}", e)))?;
        }

        let mut output = Array1::from_elem(query_vecs.shape()[0], f64::NAN);

        for (i, row) in query_vecs.rows().into_iter().enumerate() {
            let p = [row[0], row[1], row[2]];
            let result = tree
                .query(&p)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            let distance = result.distance;
            if distance <= max_distance {
                output[i] = distance
            }
        }

        Ok(output)
    }
}
