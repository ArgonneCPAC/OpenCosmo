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

Modern cosmological simulations produce vast quantities of data, often reaching petabyte scale. Working with these vasts datasets can be quite challenging and usually requires significant high performance computing (HPC) knowledge as well as access to the same resources the data are stored on. Additionally, data organization may not be obvious or well documented. For example, an individual may need to associated properties of dark matter halos in the simulation with their associated particles. These startup costs represent a significant hurdle to doing science, especially for students and individuals not familiar with HPC workloads. 

OpenCosmo solves this problem by providing a toolkit for reading, manipulating, and writing cosmological simulation data in a fully-consistent cosmological context. It provides high-level abstractions for collections of data, such as lightcones or properties+particles, that may be spread out across multiple files. OpenCosmo contains built-in support for MPI workloads, with an identical API to the one used when working with small, local datasets.

# Background

The OpenCosmo project at Argonne National Lab aims to provide scientific users with access to state-of-the-art cosmological simulations, without those users having to directly access the machines these datasets are housed on. The OpenCosmo Toolkit serves two purposes in this regime. First, it interacts directly with the raw data on a given HPC system produce results for user queries. Second, it provides a suite of high-level tools which allow users further manipulate the resulting data in the scientific terms they are familiar with. Because it works on data at any scale and has built-in MPI support, users that need to work with the full datasets can do so with the exact same API they use to prototype their analysis on a smaller dataset.

# Available Tools

In addition to standard data manipulation tools (e.g. selecting columns, filtering), the OpenCosmo Toolkit provides high-level abstraction around collections of data that are related. For example a user could open a catalog of dark matter halos, filter based on some quantity of interest, and then iterate through the halos and their associated particles in a completely automated fashion without having to do any manual crossmatching between the halo catalog and halo particles file. This capability is enabled by pre-processing the data into HDF5 files with a well-defined structure. The specification for this structure is available in the software repository.

The toolkit provides a number of other astrophysics-relevant tools including automatic unit handling (and conversions), spatial queries, and differentiation between snapshot and lightcone datasets.
