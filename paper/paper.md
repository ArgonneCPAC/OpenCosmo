---
title: 'OpenCosmo: An Analysis Toolkit for Cosmological Simulations at Petabyte Scale'
tags:
  - Python
  - astrophysics
  - cosmology
  - simulations
authors:
  - name: Patrick R. Wells
    corresponding: true 
    orcid: 0000-0000-0000-0000
    affiliation: '1'
  - name: William Hicks
    affiliation: '1'
  - name: Michael Buehlmann
    affiliation: '2'
affiliations:
 - name: High Energy Physics Division, Argonne National Laboratory, Lemont, IL 60439
   index: 1
 - name: Computational Science Division, Argone National Laboratory, Lemont, IL 60439
   index: 2
Ddate: 22 September 2025
---

# Summary
pass

# Statement of Need

Modern cosmological simulations and surveys produce vast quantities of data, often reaching petabyte scale. Working with these vasts datasets at scale usually requires high performance computing (HPC) knowledge as well as access to the same resources the data are stored on. Additionally, data organization may not be obvious or well documented. These startup costs represent a significant hurdle to doing science, especially for students and individuals not familiar with HPC workloads. 

OpenCosmo solves this problem by providing a toolkit for reading, manipulating, and writing cosmological simulation data in a fully-consistent cosmological context. It provides high-level abstractions for collections of data, such as lightcones or properties+particles, that may be spread out across multiple files. OpenCosmo contains built-in support for MPI workloads, with an identical API to the one used when working with small, local datasets.

# Background

The OpenCosmo project aims to provide scientific users with access to state-of-the-art cosmological simulations, without those users having to directly access the machines these datasets are housed on. The OpenCosmo Toolkit serves two purposes in this project. First, it interacts directly with the raw data on a given HPC system produce results for user queries. Second, it provides a suite of high-level tools which allow users further manipulate the resulting data in the scientific terms they are familiar with. Because it works on data at any scale and has built-in MPI support, users that need to work with the full datasets can do so with the exact same API they use to prototype their analysis on a smaller dataset.



# Available Tools

In addition to standard data manipulation tools (e.g. selecting columns, filtering), the OpenCosmo Toolkit provides high-level abstraction around collections of data that are related. For example a user could open a catalog of dark matter halos, filter based on some quantity of interest, and then iterate through the halos and their associated particles in a completely automated fashion. This capability is enabled by pre-processing the data into HDF5 files with a well-defined structure. The specification for this structure is available in the software repository.

The toolkit provides a number of other astrophysics-relevant tools including automatic unit handling (and conversions), spatial queries, and differentiation between snapshot and lightcone datasets. For more advanced use cases, users may provide arbitrary python functions that operate on a subset of the columns in the dataset. The toolkit can automatically evaluate the provided 

# Design Principles

The toolkit is designed around lazy evaluation and immutability. Operations on a given dataset produce new datasets (rather than modifying the existing one) and data is not read from disk or processed into user-created columns until that data is actually requested. A lazy approach is inevitable when working with very large quantities of data, but it also allows for optimization of user-defined queries. For example, consider the following query:


```python
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
fof_halo_px = oc.col("fof_halo_mass")*oc.col("fof_halo_com_vx")

ds = ds
    .with_new_columns(fof_halo_com_px = fof_halo_px)
    .filter(oc.col("fof_halo_mass") > 1e14)
    .take(1000, at="random")
    .select(("fof_halo_tag", "fof_halo_com_px"))
    .get_data()

```
This operation creates a new column containing the x component of linear momentum, queries the dataset for 1000 random halos with a mass above 10**14, and then selects the newly-created column and the column containing the halo tags. The toolkit will only load the three columns from the underlying data that are required to perform this operation. The newly-created column will only be evaluated for the 1000 halos in the sample, even though the `with_new_columns` operation preceeds the filter` and `take` operations.

