``opencosmo.open`` can now read HACC full particle snapshots (``snapshotparticles`` files, data
type ``snapshot_particles``). The file opens as a new ``ParticleCollection`` that holds the dark
matter, gas, star and AGN particles as separate datasets. Each particle type has units and can be
queried by region through its own octree index. ``ParticleCollection.bound`` and
``ParticleCollection.with_units`` apply to every particle type.
