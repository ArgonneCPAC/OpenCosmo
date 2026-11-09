``opencosmo.open`` can now read HACC full particle snapshots (``snapshotparticles`` files, data
type ``snapshot_particles``). The file opens as a new ``ParticleCollection`` that holds the dark
matter, gas, star and AGN particles as separate datasets. Each particle type has units and can be
queried by region through its own octree index. ``bound``, ``box_search``, ``with_units``,
``select`` and ``drop`` apply to every particle type, and ``with_datasets``,
``evaluate_on_dataset`` and ``get_data`` work as in ``StructureCollection``. Under MPI, every
particle type is split between ranks in the same way. A ``ParticleCollection`` can be written
with ``opencosmo.write``.
