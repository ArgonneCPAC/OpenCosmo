use pyo3::prelude::*;

#[pymodule]
pub(crate) mod spatial {
    use core::f64;

    use crate::arrays::unpack_array;
    use kiddo::dist::SquaredEuclidean;
    use kiddo::MutableKdTree;
    use numpy::ndarray::{Array1, ArrayView2};
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    fn unpack_points<'py>(array: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray2<'py, f64>> {
        let data = unpack_array::<f64, 2>(array)?;
        if data.shape()[1] != 3 {
            return Err(PyValueError::new_err("Expected an array of 3d points!"));
        }
        Ok(data)
    }

    #[pyfunction(name = "get_closest_distance_3d")]
    pub(crate) fn get_closest_distance_3d<'py>(
        py: Python<'py>,
        check_vecs: &Bound<'_, PyAny>,
        query_vecs: &Bound<'_, PyAny>,
        max_squared_distance: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        get_closest_squared_distance_3d(py, check_vecs, query_vecs, max_squared_distance)
    }

    #[pyfunction(name = "get_closest_squared_distance_3d")]
    pub(crate) fn get_closest_squared_distance_3d<'py>(
        py: Python<'py>,
        check_vecs: &Bound<'_, PyAny>,
        query_vecs: &Bound<'_, PyAny>,
        max_squared_distance: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let check_arr = unpack_points(check_vecs)?;
        let query_arr = unpack_points(query_vecs)?;
        let arr = query_closest_3d(
            check_arr.as_array(),
            query_arr.as_array(),
            max_squared_distance,
        )?;
        Ok(arr.into_pyarray(py))
    }

    fn query_closest_3d(
        check_vecs: ArrayView2<'_, f64>,
        query_vecs: ArrayView2<'_, f64>,
        max_squared_distance: f64,
    ) -> PyResult<Array1<f64>> {
        if !max_squared_distance.is_finite() || max_squared_distance < 0.0 {
            return Err(PyValueError::new_err(
                "max_squared_distance must be finite and nonnegative",
            ));
        }
        if check_vecs.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "check_vecs must contain only finite coordinates",
            ));
        }
        if query_vecs.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "query_vecs must contain only finite coordinates",
            ));
        }
        if check_vecs.nrows() > u32::MAX as usize {
            return Err(PyValueError::new_err("check_vecs contains too many points"));
        }

        let mut output = Array1::from_elem(query_vecs.nrows(), f64::NAN);
        if query_vecs.nrows() == 0 || check_vecs.nrows() == 0 {
            return Ok(output);
        }

        let mut tree: MutableKdTree<f64, 3> = MutableKdTree::default();
        for (i, row) in check_vecs.rows().into_iter().enumerate() {
            let p = [row[0], row[1], row[2]];
            tree.add(&p, i as u32)
                .map_err(|e| PyValueError::new_err(format!("Unable to add point: {:?}", e)))?;
        }

        for (i, row) in query_vecs.rows().into_iter().enumerate() {
            let p = [row[0], row[1], row[2]];
            let result = tree
                .query(&p)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            let distance = result.distance;
            if distance <= max_squared_distance {
                output[i] = distance
            }
        }

        Ok(output)
    }
}
