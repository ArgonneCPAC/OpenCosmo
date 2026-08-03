 # OpenCosmo Data Spec

This document defines the specification for data that can be read and written by the OpenCosmo Python toolkit.


## Terminology

- Dataset: In HDF5, a "dataset" is the lowest level of named organization. It contains actual data values which can be read into a numpy array. In OpenCosmo, a "dataset" is a collection of columns that can be materialized into a table. A HDF5 dataset is therefore more akin to a Column in OpenCosmo/Astropy terms. For the purposes of this spec, "dataset" will refer to hdf5 datasets unless otherwise specified.
- Group: In HDF5, a "group" is a level of organization that can contain other groups or datasets. It is akin to a directory in a filesystem.


## OpenCosmo Dataset HDF5 Layout

### Required Groups

#### data

An OpenCosmo dataset consists of at least a "data" group consisting of one or more hdf5 datasets. Each dataset will make up a single column in the final Astropy table that is provided to the user. For this reason, all datasets must be the same length. However because Astropy supports multi-dimensional columns it is possible for datasets to have more dimensions so long as the 0th dimension is the same for all datasets in the group.

**Units**

Units can be provided by attaching them as attributes to the hdf5 dataset. These units should be directly readable by astropy.units.Unit. The simplest way is to simply convert the unit to a string directly:

```python
import astropy.units as u
import astropy.cosmology.units as cu

unit = u.Mpc / cu.littleh
unit_str = str(unit)

```
Note that astropy will parse "h" as *hours"*, so be sure to use "littleh."

**Description**

A plain-text description of the column may be provided as a "description" attribute to the hdf5 dataset.

### Optional Groups

### header

If a file contains a single OpenCosmo dataset, it must also contain a header. A file with multiple OpenCosmo datasets may or may not include a header in each of the OpenCosmo datasets individually. See "The Header" below for the full contents and the "nearest enclosing header" resolution rule.

### index

The "index" group contains spatial indexing information that enables spatial querying on the data. The "index" group should contain a single attribute titled "index_type." Currently supported index types are "octtree" (for 3D snapshot data) and "healpix" (for 2D lightcone data).

The "index" group must contain one or more subgroups for each level in the sptial index. The lowest level of refinement should be titled "level_0", with higher levels of refinement labeled accordingly. For an octree, level 0 contains the entire volume of the simulation. Level 1 contains 8 octants. Level 2 contains 64 regions, and so on.

Each level group should contain a "start" and a "size" dataset." These correspond to chunks of rows in the "data" group that belong to the region with the given index (see below for an example). Because of this, the data for a given region must be contiguous.

For example, the first level of an octree contains 8 octants with index 0-7. Accessing the data from octant 3 looks something like this.

```python

start = file["index"]["level_1"]["start"][3]
size = file["index"]["level_1"]["size"][3]
columne = file["data"]["some_column"][start: start + size]
```

The octree uses a z-order curve to assign octants to indices. In level 2 of the octree, the octants that subdivied the 0th octant in level one will be indexed 0->7, while the subidivisions of the 1st octant will be indexex 8->15, and so on.

For healpix index, pixels use "nested" ordering.

### data_linked

The "data_linked" group contains an index into other OpenCosmo datasets that contain complimentary information. For example, a group with halo properties may contain an index into a halo particles group which specifies where the particles for the given halo can be found. This linking information can be used by the library to retrieve auxillary data as needed.

Within the "data_linked" group are one or more datasets that specify the indices. These datasets should be named with some unique identifier, followed by a suffix that specifies the type of link. The length of these these datasets must be the same as the length of the datasets in the "data" group. 

For example a "halo_properties/data_linked" group could contain "haloparticles_start" and "haloparticles_size." Each row specifies a range of rows in a haloparticles group in exactly the same way as the spatial index specifies a range of rows. If there are no rows in the link target corresponding to the given row in the link source, the size should be set to 0.

For rows that have a one-to-one correspondence to rows in another group, a single dataset with the suffix "idx" should be used. These values will be used directly to index the link target. If a given row in the link source has no corresponding row in the target, the idx should be set to -1.

Because it is possible to have many files with a given data type, It is recommended that link names include some sort of UUID or other unique identifier that will be varied across files of a single data type.


## Multiple OpenCosmo Datasets in a File

A single file may contain multiple OpenCosmo Datasets partitioned into groups. For example, a single file containing both halo properties and halo particles would be structured as follows

```text
/halo_properties
    /data
    /data_linked
    /index
    /header
/halo_particles
    /data
    /index
    /header
/header
```

The "halo_properties" group contains a "data_linked" group which allows the toolkit to associate rows in "halo_properties" to rows in "halo_particles."


## The Header

The "header" group contains information about the file and the OpenCosmo datasets it governs: cosmology, simulation parameters, and a `file` block of per-dataset metadata. The library reads it into an `OpenCosmoHeader` (`python/opencosmo/header.py`) whose `file` field is a `FileParameters` model (`python/opencosmo/dtypes/file.py`). The fields that drive file identification are:

- **`data_type`** — one of `galaxy_properties`, `galaxy_particles`, `halo_properties`, `halo_profiles`, `halo_particles`, `synthetic_galaxies`, `healpix_map`. (`diffsky_fits` is normalized to `synthetic_galaxies` on read.)
- **`is_lightcone`** — `True` when the dataset covers a lightcone (2D sky) rather than a 3D snapshot volume.
- **`step`** — the redshift step for stacked/redshift-split data; `None` for a single-step snapshot.
- **`redshift`**, **`region`**, **`unit_convention`** — additional per-dataset metadata.

### Nearest enclosing header

A header at `/header` governs the root scope `/`; a header at `/scidac1/header` governs the `/scidac1` scope. The governing header of any `/data` group is the **nearest enclosing header** — the deepest header whose group is an ancestor of that `/data` group, walking up the group's own ancestry. A file that stores one header per dataset (a structure collection: `/halo_properties/header`, `/halo_particles/header`, ...) and a file with a single top-level `/header` both resolve correctly under this rule.

Identification never depends on header *content* equality. Two independent simulations may legitimately share identical cosmology and simulation parameters; scopes are told apart by their in-file group path (`header_path`), not by comparing header values.


## File and Collection Types

`opencosmo.open()` inspects file metadata (a "layout" — see `python/opencosmo/io/discover.py`) and dispatches to one of the types below. Single-file datasets and collections are recognized by a small registry of **specs** in `python/opencosmo/io/specs.py`; the four spec names are `dataset`, `healpix_map`, `lightcone`, and `structure_collection`. A simulation collection is not a spec — it is recognized structurally (see below) and assembled from per-scope children.

Identification uses only metadata-derived signals — never live reads of column values:

- `data_type` — the header field above.
- `is_lightcone` — the header field above.
- presence of a `/data_linked` group carrying link targets.
- whether a group is a "properties" group (`data_type` in `halo_properties` / `galaxy_properties`).
- the number of distinct governing header scopes / top-level containers across the open set.

Specs are tried in a fixed precedence order (most-constrained first): `structure_collection`, `healpix_map`, `lightcone`, `dataset`. The first whose match predicate is true wins. This order matters — a structure collection would also satisfy weaker predicates, so it is tested first.

### dataset (`dataset` spec)

A single OpenCosmo dataset. **Signal:** exactly one `/data` group, and that group is neither a lightcone (`is_lightcone` false) nor a healpix map (`data_type != "healpix_map"`). Opens as a `Dataset`.

A lone properties file that carries `/data_linked` still opens as a `dataset` (or a `lightcone`), not a structure collection: its links reference child datasets that are not present in the open set. The collection only comes into being once a linked child type is opened alongside it.

### healpix map (`healpix_map` spec)

A full-sky or partial-sky HEALPix map. **Signal:** exactly one `/data` group whose `data_type == "healpix_map"`. Opens as a `HealpixMap`. Its `/index` group uses `index_type == "healpix"` with nested pixel ordering.

### lightcone (`lightcone` spec)

Sky-coverage data indexed by HEALPix pixels, optionally split across redshift steps. **Signal:** one or more `/data` groups, **all** with `is_lightcone` true, and **all** sharing a single `data_type`. Opens as a `Lightcone`.

The single-`data_type` requirement is what distinguishes a lightcone from a lightcone *structure* collection: a structure collection always carries more than one `data_type`, so it is caught by the earlier `structure_collection` spec before this one is reached.

A redshift-split lightcone writes each step under a `<step>_<name>` subgroup (e.g. `/halo_properties/600_data`). These step subgroups are collapsed to one logical dataset during identification, so many steps of one dataset are not mistaken for many independent datasets.

### structure collection (`structure_collection` spec)

Parent–child datasets linked via `/data_linked` — e.g. halo properties plus halo particles/profiles. **Signal:** at least one properties group (`data_type` in `halo_properties`/`galaxy_properties`) that carries a `/data_linked` group, **and** more than one distinct `data_type` present in the open set. Opens as a `StructureCollection`.

The "more than one `data_type`" requirement ensures at least one linked *child* type is actually being opened; a properties file on its own opens as a plain dataset (see above). Across redshift steps, a structure collection must present an identical set of `data_type`s at every step, and every step must contain a properties-with-`/data_linked` group; these are verified structurally (cross-file), without reading link indices or row counts.

### simulation collection (structural, not a spec)

Multiple independent datasets or collections from one or more simulations, e.g. snapshots across redshifts, or two simulations (`/scidac1/...`, `/scidac2/...`) in one file. This is **not** a spec: `group_by_scope` (`python/opencosmo/io/specs.py`) detects it and splits the layout into one single-scope sub-layout per simulation.

**Signal:** the open set does not resolve to a single spec across all its groups, *or* its datasets are nested under more than one top-level container. Either case is split by top-level container; each resulting scope is matched and built through the ordinary single-scope path above, and the children are wrapped in a `SimulationCollection`. (A `SimulationCollection` is therefore only ever *assembled* from already-built children — it is never opened through a spec builder.)

A nested simulation collection of structure collections looks like this on disk (one header per simulation scope, and within each scope the ordinary structure-collection layout):

```text
/scidac1
    /header
    /halo_properties
        /data
        /data_linked
        /index
    /halo_particles
        /data
        /index
/scidac2
    /header
    /halo_properties
        /data
        /data_linked
        /index
    /halo_particles
        /data
        /index
```
