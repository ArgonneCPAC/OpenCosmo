# # # OpenCosmo Data Spec

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

The octree uses a z-order curve to assign octants to indices. In level 2 of the octree, the octants that subdivied the 0th octant in level one will be indexed 0->7, while the octants corresponding to the 1st octant 

