opencosmo 1.5.0b1 (2026-11-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bugfixes
^^^^^^^^

- Columns added through ``evaluate``, ``with_new_columns`` or ``select`` can now be named ``mode``, ``descriptions``, ``allow_overwrite`` or ``dataset`` without being mistaken for an option of the method that adds them.
- Derived columns added to linked datasets in a ``StructureCollection`` are now still
  cached during iteration after ``with_units`` or ``with_new_columns`` on a nested
  collection. Previously these calls caused such columns to be recomputed after
  iterating with ``objects()``.
- Filtering a HEALPix map after ``with_resolution`` now keeps each value paired with its pixel. Previously the downgraded map lost track of which pixels a filter removed.
- Fixed header comparisons that depended on whether a file layout came from the discovery cache, and stopped lightcone headers from leaking lightcone parameters into later snapshot headers.
- Iterating a dataset that produces no chunks now ends cleanly instead of raising ``RuntimeError``. The internal row iterator previously raised ``StopIteration`` inside a generator, which PEP 479 converts into ``RuntimeError``.
- Opening a file whose header has no cosmology block, or more than one, now raises a ``ValueError`` instead of a bare ``IndexError`` or silently using the first block.
- Opening a lightcone with a cone or skybox region under MPI no longer raises ``AttributeError``. Opening a snapshot whose header has no box size now raises a clear ``ValueError``.
- Partial-sky ``HealpixMap.get_data("healpix")`` output now masks the pixels outside the map's coverage, rather than the pixels inside it, and fills them with ``healpy.UNSEEN`` instead of 0. Each column also gets its own mask.
- Reopening a partial-sky HEALPix map written by OpenCosmo now pairs each value with the correct pixel. Previously the stored pixels were read back in arbitrary order.
- Selecting columns on a sorted ``HealpixMap`` no longer leaks the sort column into ``columns`` or the output maps.
- ``HealpixMap.evaluate(insert=False)`` with a function that returns ``None`` now returns ``{name: None}``, as ``Dataset.evaluate`` does, instead of raising.
- ``HealpixMap.full_sky`` now reflects the pixels each rank holds. Previously every rank of a partitioned full-sky map reported full coverage under MPI.
- ``SimulationCollection.with_new_columns`` now realigns a matched collection before adding
  columns, so per-dataset arrays sized to the matched rows are accepted after operations
  such as ``filter`` or ``take`` on the active source.
- ``SimulationCollection`` now accepts lightcones as members, and using a collection of
  structure collections as a context manager no longer raises an ``AttributeError``.
  ``SimulationCollection.match`` also keeps the collection's key order instead of moving
  the matched dataset to the front.
- ``StructureCollection.evaluate`` with a nested dataset path (for example
  ``dataset="galaxies.star_particles"``) now adds the result to the nested
  ``galaxies`` collection. Previously it was stored under a new
  ``"galaxies.star_particles"`` key and the nested collection was left unchanged.


New Features
^^^^^^^^^^^^

- :py:class:`Dataset <opencosmo.Dataset>` objects now track which operations have been performed on them since they were originally produced.
- :py:meth:`SimulationCollection.match <opencosmo.SimulationCollection.match>` now supports structure collections.
- Add ``SimulationCollection.take_rows``. On a matched collection, rows are taken from the
  active source and matching rows are retained in the other datasets.
- Added ``opencosmo.serde`` messages for ``Lightcone`` transformations (``filter``,
  ``select``, ``drop``, ``sort_by``, ``take``, ``take_range``, ``take_rows``,
  ``bound``, ``with_new_columns``, ``with_units``, ``with_redshift_range`` and
  ``pixel_search``), generated from the method signatures.
  ``opencosmo.serde.execute_message`` now accepts a ``Lightcone`` target, and
  variadic keyword names that collide with explicit method parameters are rejected
  when a message is parsed.
- Added ``opencosmo.serde`` messages for ``StructureCollection`` transformations
  (``filter``, ``select``, ``drop``, ``sort_by``, ``take``, ``take_range``, ``take_rows``,
  ``bound``, ``with_new_columns``, ``with_units``, ``with_datasets``,
  ``with_redshift_range`` and ``pixel_search``), generated from the method signatures.
  ``select``, ``drop`` and ``with_units`` accept nested per-dataset specifications.
  ``opencosmo.serde.execute_message`` now accepts a ``StructureCollection`` target, and
  variadic keyword names that collide with explicit method parameters are rejected
  when a message is parsed.
- Added ``opencosmo.serde``, which describes ``Dataset`` transformations as validated,
  JSON-serializable messages generated from the ``Dataset`` method signatures.
  ``opencosmo.serde.execute_message`` parses a decoded JSON message for an OpenCosmo
  object and applies it, returning a ``MessageError`` when the message is invalid or
  unsupported.
- Lightcone catalogs can now be opened with HEALPix maps and produce square FITS cutouts around catalog objects.
- The sigma_8 parameter for a dataset is now attached to the cosmology meta, and can be accessed with `ds.cosmology.meta["sigma_8"]
- ``StructureCollection.evaluate`` can now take a function whose arguments are column names. The
  dataset each column belongs to is inferred, and a column found in more than one dataset raises a
  ``ValueError``. The explicit form, with dataset names as arguments, is unchanged.
- ``opencosmo.open`` can now read HACC full particle snapshots (``snapshotparticles`` files, data
  type ``snapshot_particles``). The file opens as a new ``ParticleCollection`` that holds the dark
  matter, gas, star and AGN particles as separate datasets. Each particle type has units and can be
  queried by region through its own octree index. ``bound``, ``box_search``, ``with_units``,
  ``select`` and ``drop`` apply to every particle type, and ``with_datasets``,
  ``evaluate_on_dataset`` and ``get_data`` work as in ``StructureCollection``. Under MPI, every
  particle type is split between ranks in the same way. A ``ParticleCollection`` can be written
  with ``opencosmo.write``.


Improvements
^^^^^^^^^^^^

- :py:meth:`opencosmo.open` can now handles multiple :py:class:`StructureCollections <opencosmo.StructureCollection>` which should be packaged together as a :py:class:`SimulationCollection <opencosmo.SimulationCollection>`
- Reads from hdf5 are now planned to balance performance and memory usage.


Miscellaneous
^^^^^^^^^^^^^

- Cleared the ruff 0.17 lint backlog across the codebase (import sorting, typing modernization, comprehension and control-flow simplifications) with no behavior change, unblocking the dependency bump that upgrades ruff.
- Test data is now cached from ``main`` by a dedicated workflow, so pull requests without
  access to repository secrets (such as Dependabot PRs) can restore it.
- The dependency on `rustworkx` has been removed by implementing core dependency solving algorithms manually.
- The project now uses ty instead of mypy for static type checking. Run `uv run ty check`.
- Tightened the type annotations of ``Lightcone`` transformation methods (``Literal``
  for ``at`` and ``convention``, non-negative row counts, explicit return types), and
  removed the unused ``**kwargs`` from ``Lightcone.filter``.
- Tightened the type annotations of ``StructureCollection`` transformation methods
  (``Literal`` for ``mode``, ``at`` and ``convention``, explicit return types, and
  typed nested selection, drop and unit parameters).



