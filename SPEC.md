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

If a file contains a single OpenCosmo dataset, it must also contain a header. A file with multiple OpenCosmo datasets may or may not include a header in each of the OpenCosmo datasets individually. See below for more information.

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
/halo_particles
    /data
    /index
/header
```

Note the presence of the "header" group, which will be discussed below. The "halo_properties" group contains a "data_linked" group which allows the toolkit to associate rows in "halo_properties" to rows in "halo_particles."

## The Header

The "header" group contains information about the file and the OpenCosmo datasets it contains. In general, this header will be mostly identical for any set of OpenCosmo datasets which are drawn from a single simulation (more documentation to come)...




