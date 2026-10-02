Regions
-------

Regions can be used to spatially query datasets and collections. Regions are constructed with :py:meth:`opencosmo.make_box` or :py:meth:`opencosmo.make_cone`.

.. autofunction:: opencosmo.make_box 
.. autofunction:: opencosmo.make_cone
.. autofunction:: opencosmo.make_skybox

.. autoclass:: opencosmo.spatial.BoxRegion
   :members:
   :member-order: bysource

.. autoclass:: opencosmo.spatial.ConeRegion
   :members:
   :member-order: bysource

.. autoclass:: opencosmo.spatial.HealpixRegion
   :members:
   :member-order: bysource

Query behavior
--------------

Spatial queries use the following conventions:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Behavior
   * - Point containment
     - Box, cone, and skybox boundaries are excluded.
   * - Box relationships
     - Box containment includes equal boundaries, and boxes that touch are considered
       intersecting when planning spatial partitions.
   * - Snapshot coverage
     - A wholly disjoint query produces an empty dataset. A query extending partly
       beyond the current dataset region executes with a warning. Snapshot coordinates
       are not wrapped periodically.
   * - Right ascension
     - Skyboxes may wrap across zero degrees. Queries near the celestial poles are
       supported.
   * - Catalog HEALPix queries
     - Catalog indexes distinguish pixels contained by the query from boundary pixels.
       Rows in boundary pixels receive an exact coordinate check.
   * - Map HEALPix queries
     - By default, :meth:`~opencosmo.HealpixMap.bound` selects pixels whose centers are
       in a cone. Passing ``inclusive=True`` selects pixels that overlap the cone.
   * - Index ranges
     - Row and pixel ranges are half-open: ``[start, start + size)``. Zero-sized input
       ranges are valid empty ranges and are omitted from generated selections.

Internal nearest-neighbor queries use squared Euclidean distances and include matches
whose squared distance equals the threshold. When an index remapping source contains a
duplicate value, the last occurrence determines the mapped position.
