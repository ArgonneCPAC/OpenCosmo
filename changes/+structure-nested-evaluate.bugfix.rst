``StructureCollection.evaluate`` with a nested dataset path (for example
``dataset="galaxies.star_particles"``) now adds the result to the nested
``galaxies`` collection. Previously it was stored under a new
``"galaxies.star_particles"`` key and the nested collection was left unchanged.
