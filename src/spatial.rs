use pyo3::prelude::*;

#[pymodule]
pub(crate) mod spatial {
    use core::f64;

    use crate::arrays::unpack_array;
    use kiddo::dist::SquaredEuclidean;
    use kiddo::MutableKdTree;
    use numpy::ndarray::{Array1, ArrayView2};
    use numpy::{IntoPyArray, PyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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

    struct OctreeQueryResult {
        pub(crate) contained: Vec<Array1<usize>>,
        pub(crate) remainder: (Array1<usize>, Array1<usize>),
    }

    #[pyfunction]
    fn get_octree_indices<'py>(
        py: Python<'py>,
        box_size: f64,
        query_bounds: (f64, f64, f64, f64, f64, f64),
        max_level: usize,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)>> {
        let query_bounds = [
            query_bounds.0 / box_size,
            query_bounds.1 / box_size,
            query_bounds.2 / box_size,
            query_bounds.3 / box_size,
            query_bounds.4 / box_size,
            query_bounds.5 / box_size,
        ];
        let result = visit_octant(0, max_level, (0, 0, 0), query_bounds);
        let mut level_contains = result.contained;
        let remainder = result.remainder;
        let mut output = Vec::new();

        for level_result in level_contains.drain(..) {
            output.push((
                PyArray::from_owned_array(py, level_result),
                PyArray::zeros(py, [0], false),
            ))
        }
        output.push((
            PyArray::from_owned_array(py, remainder.0),
            PyArray::from_owned_array(py, remainder.1),
        ));
        Ok(output)
    }

    fn visit_octant(
        level: usize,
        max_level: usize,
        index: (usize, usize, usize),
        query_bounds: [f64; 6],
    ) -> OctreeQueryResult {
        let nside = 1usize << level;
        let octant_size = 1.0 / nside as f64;
        let octant_bounds = [
            index.0 as f64 * octant_size,
            (index.0 + 1) as f64 * octant_size,
            index.1 as f64 * octant_size,
            (index.1 + 1) as f64 * octant_size,
            index.2 as f64 * octant_size,
            (index.2 + 1) as f64 * octant_size,
        ];

        let intersects = query_bounds[0] < octant_bounds[1]
            && query_bounds[1] > octant_bounds[0]
            && query_bounds[2] < octant_bounds[3]
            && query_bounds[3] > octant_bounds[2]
            && query_bounds[4] < octant_bounds[5]
            && query_bounds[5] > octant_bounds[4];
        if !intersects {
            return empty_octree_query_result(max_level);
        }

        let octant_index = get_z_order_index(index, level);
        let contained = query_bounds[0] <= octant_bounds[0]
            && query_bounds[1] >= octant_bounds[1]
            && query_bounds[2] <= octant_bounds[2]
            && query_bounds[3] >= octant_bounds[3]
            && query_bounds[4] <= octant_bounds[4]
            && query_bounds[5] >= octant_bounds[5];
        if level == max_level {
            let mut output = empty_octree_query_result(max_level);
            let remainder = if contained {
                &mut output.remainder.0
            } else {
                &mut output.remainder.1
            };
            *remainder = Array1::from_vec(vec![octant_index]);
            return output;
        }

        if contained {
            let mut output = empty_octree_query_result(max_level);
            output.contained[level] = Array1::from_vec(vec![octant_index]);
            return output;
        }

        let mut contained_output: Vec<Vec<usize>> = (0..max_level).map(|_| Vec::new()).collect();
        let mut remainder_contained = Vec::new();
        let mut remainder_overlapping = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let child_output = visit_octant(
                        level + 1,
                        max_level,
                        (2 * index.0 + x, 2 * index.1 + y, 2 * index.2 + z),
                        query_bounds,
                    );
                    for (indices, child_indices) in
                        contained_output.iter_mut().zip(child_output.contained)
                    {
                        indices.extend(child_indices);
                    }
                    remainder_contained.extend(child_output.remainder.0);
                    remainder_overlapping.extend(child_output.remainder.1);
                }
            }
        }
        OctreeQueryResult {
            contained: contained_output.into_iter().map(Array1::from_vec).collect(),
            remainder: (
                Array1::from_vec(remainder_contained),
                Array1::from_vec(remainder_overlapping),
            ),
        }
    }

    fn empty_octree_query_result(max_level: usize) -> OctreeQueryResult {
        OctreeQueryResult {
            contained: (0..max_level).map(|_| Array1::default(0)).collect(),
            remainder: (Array1::default(0), Array1::default(0)),
        }
    }

    fn get_z_order_index(index: (usize, usize, usize), level: usize) -> usize {
        let mut output = 0;
        for bit in 0..level {
            output |= ((index.0 >> bit) & 1) << (3 * bit);
            output |= ((index.1 >> bit) & 1) << (3 * bit + 1);
            output |= ((index.2 >> bit) & 1) << (3 * bit + 2);
        }
        output
    }

    #[pyfunction]
    pub(crate) fn partition_bounding_box<'py>(
        box_indices: &Bound<'_, PyAny>,
        box_size: f64,
        level: usize,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        let index = unpack_array::<i64, 1>(box_indices)?;
        let mut bounds = (0., box_size, 0., box_size, 0., box_size);
        for octant_bounds in index
            .as_array()
            .iter()
            .map(|&i| get_bounds_from_index(i as usize, level, box_size))
        {
            bounds = (
                octant_bounds[0].min(bounds.0),
                octant_bounds[1].max(bounds.1),
                octant_bounds[2].min(bounds.2),
                octant_bounds[3].max(bounds.3),
                octant_bounds[4].min(bounds.4),
                octant_bounds[5].max(bounds.5),
            )
        }

        Ok(bounds)
    }

    fn get_3d_index(z_order_index: usize, level: usize) -> (usize, usize, usize) {
        let mut index = (0, 0, 0);
        for bit in 0..level {
            index.0 |= ((z_order_index >> (3 * bit)) & 1) << bit;
            index.1 |= ((z_order_index >> (3 * bit + 1)) & 1) << bit;
            index.2 |= ((z_order_index >> (3 * bit + 2)) & 1) << bit;
        }
        index
    }

    fn get_bounds_from_index(z_order_index: usize, level: usize, box_size: f64) -> [f64; 6] {
        let index_3d = get_3d_index(z_order_index, level);
        let base: i64 = 2;
        let octant_size = box_size / (base.pow(level as u32) as f64);
        [
            octant_size * index_3d.0 as f64,
            octant_size * (index_3d.0 + 1) as f64,
            octant_size * (index_3d.1) as f64,
            octant_size * (index_3d.1 + 1) as f64,
            octant_size * index_3d.2 as f64,
            octant_size * (index_3d.2 + 1) as f64,
        ]
    }
}
