use pyo3::prelude::*;

#[pymodule]
pub(crate) mod graph {
    use petgraph::algo::{is_cyclic_directed, toposort};
    use petgraph::graph::{DiGraph, NodeIndex};
    use pyo3::exceptions::PyValueError;
    use std::collections::HashMap;
    use std::sync::Arc;

    use pyo3::prelude::*;

    #[pyclass]
    struct DependencyGraphRegistry {
        pub(crate) reg: HashMap<i64, Arc<DependencyGraph>>,
    }

    #[pymethods]
    impl DependencyGraphRegistry {
        #[staticmethod]
        fn new() -> Self {
            return DependencyGraphRegistry {
                reg: HashMap::new(),
            };
        }

        fn add_new_graph(&mut self, id: i64, nodes: HashMap<String, Vec<String>>) -> PyResult<()> {
            add_graph(self, id, nodes)
        }
        fn add_child_graph(&mut self, parent_id: i64, child_id: i64) -> PyResult<()> {
            add_child(self, parent_id, child_id)
        }
        fn get_order(&self, id: i64) -> PyResult<Vec<String>> {
            let graph = self
                .reg
                .get(&id)
                .ok_or(PyValueError::new_err("Unknown graph id!"))?;
            let sort = toposort(&graph.graph, None).unwrap(); // Checked at build time
            Ok(sort
                .iter()
                .map(|idx| graph.uuid_map.get(idx).unwrap().clone())
                .collect())
        }
    }

    struct DependencyGraph {
        pub(crate) graph: DiGraph<String, ()>,
        pub(crate) uuid_map: HashMap<NodeIndex, String>,
    }

    fn make_dependency_graph(nodes: HashMap<String, Vec<String>>) -> PyResult<DependencyGraph> {
        let mut graph = DiGraph::<String, ()>::new();
        let mut uuid_map: HashMap<String, NodeIndex<u32>> = HashMap::new();
        for node_name in nodes.keys() {
            uuid_map.insert(node_name.clone(), graph.add_node(node_name.clone()));
        }
        for (node_name, connections) in nodes.iter() {
            for con in connections.iter() {
                if let Some(&con_id) = uuid_map.get(con) {
                    _ = graph.add_edge(*uuid_map.get(node_name).unwrap(), con_id, ());
                } else {
                    return Err(PyValueError::new_err(
                        "Found an invalid edge in the dependency graph!",
                    ));
                }
            }
        }
        if is_cyclic_directed(&graph) {
            return Err(PyValueError::new_err("Graph contains a dependency cycle!"));
        }

        Ok(DependencyGraph {
            graph: graph,
            uuid_map: uuid_map.into_iter().map(|(k, v)| (v, k)).collect(),
        })
    }

    fn add_child(reg: &mut DependencyGraphRegistry, parent_id: i64, child_id: i64) -> PyResult<()> {
        if let Some(parent_graph) = reg.reg.get(&parent_id) {
            if reg.reg.contains_key(&child_id) {
                return Err(PyValueError::new_err(
                    "Tried to insert a duplicate graph ID!",
                ));
            }
            _ = reg.reg.insert(child_id, parent_graph.clone())
        } else {
            return Err(PyValueError::new_err("Unknown parent id!"));
        }
        Ok(())
    }

    fn add_graph(
        reg: &mut DependencyGraphRegistry,
        id: i64,
        nodes: HashMap<String, Vec<String>>,
    ) -> PyResult<()> {
        if reg.reg.contains_key(&id) {
            return Err(PyValueError::new_err("Tried to add a duplicated ID!"));
        }
        let graph = make_dependency_graph(nodes)?;
        reg.reg.insert(id, Arc::new(graph));

        Ok(())
    }
}
