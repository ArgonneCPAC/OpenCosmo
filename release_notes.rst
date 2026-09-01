opencosmo 1.4.0b1 (2026-09-01)
==============================

Bugfixes
--------

- Opening particle data on its own (e.g. ``oc.open("haloparticles.hdf5")``) now raises a clear ``ValueError`` instead of silently returning a bogus ``SimulationCollection``. Particle datasets must be opened together with the properties dataset that links to them, as a ``StructureCollection``. (#257)
- Fix a bug that could cause writes to fail fora a dataset which is sorted by a column which was later dropped. (#265)
- Fixed a crash when opening a lightcone structure collection under ``mpi_mode="redshift"`` with more ranks than redshift steps, where a linked source that derives its ``redshift`` column from comoving distance (e.g. galaxy properties) would fail on the surplus zero-length ranks with ``ValueError: Iteration of zero-sized operands is not enabled``. The redshift derivation now short-circuits on empty input instead of invoking astropy's ``z_at_value`` solver. (#272)
- Optimizes logic for building and re-building datasets when working with :py:class:`StructureCollection <opencosmo.StructureCollection>`s that contain :py:class:`Lightcone <opencosmo.Lightcones>` to avoid potential memory overloading issues. (#275)
- Fix a bug that could cause :my:meth:`reduce <opencosmo.analysis.reduce>` to fail if used on a simulation collection with keys that are not kwarg compatible.


Documentation
-------------

- Rewrote ``SPEC.md`` to document every file and collection type (dataset, healpix map, lightcone, structure collection, and simulation collection, including nested simulation collections) and the metadata-only signals used to identify each — ``data_type``, ``is_lightcone``, presence of ``/data_linked``, the properties-group check, and the number of distinct header scopes — cross-referencing the four spec names in ``io/specs.py`` and the structural ``group_by_scope`` rule for simulation collections. (#256)


New Features
------------

- :py:meth:`Dataset.select <opencosmo.Dataset.select>` can now retrieve scalar summary statistics directly. Pass scalar expressions as keyword arguments — ``get_data()`` returns an astropy Quantity for a single scalar or a dict of Quantities for multiple:

  .. code-block:: python

     min_mass = ds.select(min_mass=oc.col("fof_halo_mass").min()).get_data()

     stats = ds.select(
         min_mass=oc.col("fof_halo_mass").min(),
         max_mass=oc.col("fof_halo_mass").max(),
     ).get_data()

  Scalar reductions respect any prior ``filter()`` or ``bound()`` calls. Scalar and column selections cannot be mixed in a single ``select()`` call. (252.1)
- :py:meth:`Dataset.select <opencosmo.Dataset.select>`, :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>`, and :py:meth:`Dataset.filter <opencosmo.Dataset.filter>` (and the equivalents on :py:class:`Lightcone <opencosmo.Lightcone>` and :py:class:`StructureCollection <opencosmo.StructureCollection>`) now accept a ``mode`` keyword argument. The default ``mode="global"`` combines scalar reductions across all ranks under MPI before they are used, so every rank ends up with the same value. Pass ``mode="local"`` to restrict the reduction to each rank's own chunk. This applies to top-level scalar selections, scalar reductions nested inside derived column expressions, and scalars used in filter masks:

  .. code-block:: python

     m = oc.col("fof_halo_mass")

     # Scalar selection — defaults to the cross-rank global value
     global_min = ds.select(min_mass=m.min()).get_data()

     # Per-rank scalar
     local_min = ds.select(min_mass=m.min(), mode="local").get_data()

     # Derived column normalized against the global mean and std
     ds = ds.with_new_columns(zscore=(m - m.mean()) / m.std())

     # Filter against a globally-computed threshold
     ds = ds.filter(m > m.mean())

  ``mode`` has no effect on plain column selections, on expressions without scalar reductions, or when not running under MPI. (252.2)
- :py:meth:`opencosmo.col` expressions now support scalar reduction methods: ``.mean()``, ``.min()``, ``.max()``, ``.std()``, ``.var()``, ``.median()``, ``.sum()``, and ``.quantile(q)``. Scalar reductions can be used in column arithmetic (e.g. normalization) and in filter expressions:

  .. code-block:: python

     m = oc.col("fof_halo_mass")

     # Normalize a column
     ds = ds.select("*", scaled=(m - m.min()) / (m.max() - m.min()))

     # Filter relative to a data-driven threshold
     ds = ds.filter(m < m.mean()) (#252)
- Added a ``mpi_mode`` argument to :py:func:`opencosmo.open`. The default, ``"spatial"``, preserves the existing behavior where every MPI rank opens every file and reads a spatial sub-partition. The new ``"redshift"`` mode splits the redshift-ordered steps into contiguous chunks of roughly-equal data volume and gives each rank one chunk, so each file is opened by exactly one rank and each rank owns a continuous redshift range. This eliminates the redundant filesystem metadata traffic that made spatial splitting pathological for many-file lightcones: only rank 0 reads every header (to build the plan), and each rank then constructs targets solely for its assigned files. Ranks that receive no files still hold a full-schema, zero-length dataset, so ``select``/``filter`` and scalar reductions behave identically on every rank. Redshift mode also distributes lightcone **structure collections**: when each redshift step is described by several linked files (halo properties plus their particles and profiles), all of a step's linked files travel together to one rank, and every rank builds a matching :py:class:`~opencosmo.StructureCollection`. Nested (Diffsky step-then-type) lightcones automatically fall back to spatial distribution. (#272)
- Add :py:meth:`SimulationCollection.match <opencosmo.SimulationCollection.match>`, which allows matching of objects between
  simulations. (#281)
- :py:meth:`StructureCollection.select <opencosmo.StructureCollection.select>` and :py:meth:`StructureCollection.drop <opencosmo.StructureCollection.drop>` can now infer which columns belong to which datasets, allow select calls that look like dataset select calls.
- Add a :py:meth:`gather <opencosmo.analysis.gather>` method, which concatenates data from multiple ranks into a single array and (optionally) passes them to a user-defined plotting function.
- Parallel writes (with parallel hdf5) now have compression enabled.


Improvements
------------

- Significant rewrite of the logic underpinning `open` to improve readability and extensability, and minimize the number of metadata reads in highly parallel environment. User-facing interface is unchanged. (#274)
- Extending the pyxsim integration to use source models other than just CIESourceModel


Miscellaneous
-------------

- Add references to OpenCosmo paper to documentation and README.
- Replaced "deprecated" library with a custom "deprecated" wrapper.
- Update transient dependencies to clear security issues discovered by Depandabot.



