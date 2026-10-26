use pyo3::prelude::*;

#[pymodule]
pub(crate) mod graph {
    use petgraph::algo::{is_cyclic_directed, toposort};
    use petgraph::graph::{DiGraph, NodeIndex};
    use petgraph::Direction;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use std::collections::{HashMap, HashSet};
    use uuid::Uuid;

    use pyo3::prelude::*;

    #[pyclass(frozen)]
    struct DependencyGraph {
        graph: DiGraph<Uuid, ()>,
        uuid_map: HashMap<Uuid, NodeIndex>,
    }

    #[pymethods]
    impl DependencyGraph {
        fn num_nodes(&self) -> usize {
            self.graph.node_count()
        }

        #[pyo3(signature = (required_nodes=None))]
        fn order(&self, required_nodes: Option<HashSet<Uuid>>) -> PyResult<Vec<Uuid>> {
            if let Some(nodes) = required_nodes {
                return self.get_all_required(nodes);
            }
            let order = toposort(&self.graph, None)
                .map_err(|_| PyRuntimeError::new_err("Graph contains a dependency cycle!"))?;
            Ok(order.into_iter().map(|idx| self.graph[idx]).collect())
        }

        fn get_all_required(&self, nodes: HashSet<Uuid>) -> PyResult<Vec<Uuid>> {
            let mut roots = nodes
                .iter()
                .map(|node| {
                    self.uuid_map
                        .get(node)
                        .copied()
                        .ok_or_else(|| PyValueError::new_err("Unknown column uuid"))
                })
                .collect::<PyResult<Vec<NodeIndex>>>()?;
            // Reverse-sorted so roots are popped in ascending index order, keeping output deterministic.
            roots.sort_unstable_by(|a, b| b.cmp(a));

            // Iterative post-order DFS over incoming edges: a node is emitted only after
            // all of its dependencies, which yields a topological order.
            let mut visited: HashSet<NodeIndex> = HashSet::with_capacity(roots.len());
            let mut order: Vec<NodeIndex> = Vec::new();
            let mut stack: Vec<(NodeIndex, bool)> = roots.into_iter().map(|r| (r, false)).collect();
            while let Some((idx, expanded)) = stack.pop() {
                if expanded {
                    order.push(idx);
                    continue;
                }
                if !visited.insert(idx) {
                    continue;
                }
                stack.push((idx, true));
                stack.extend(
                    self.graph
                        .neighbors_directed(idx, Direction::Incoming)
                        .filter(|n| !visited.contains(n))
                        .map(|n| (n, false)),
                );
            }

            Ok(order.into_iter().map(|idx| self.graph[idx]).collect())
        }
    }

    /// Build a dependency graph from a mapping of producer UUID to the UUIDs it requires.
    #[pyfunction]
    fn make_dependency_graph(nodes: HashMap<Uuid, HashSet<Uuid>>) -> PyResult<DependencyGraph> {
        // Sorted so node indices, and therefore traversal order, do not depend on hash order.
        let mut node_ids: Vec<Uuid> = nodes.keys().copied().collect();
        node_ids.sort_unstable();

        let mut graph = DiGraph::<Uuid, ()>::with_capacity(node_ids.len(), 0);
        let uuid_map: HashMap<Uuid, NodeIndex> = node_ids
            .iter()
            .map(|&id| (id, graph.add_node(id)))
            .collect();

        for id in node_ids.iter() {
            let target = uuid_map[id];
            let mut deps: Vec<Uuid> = nodes[id].iter().copied().collect();
            deps.sort_unstable();
            for dep in deps {
                let Some(&source) = uuid_map.get(&dep) else {
                    return Err(PyValueError::new_err(
                        "Found an invalid edge in the dependency graph!",
                    ));
                };
                graph.add_edge(source, target, ());
            }
        }
        if is_cyclic_directed(&graph) {
            return Err(PyValueError::new_err("Graph contains a dependency cycle!"));
        }

        Ok(DependencyGraph { graph, uuid_map })
    }
}
