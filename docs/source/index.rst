.. image:: ../../branding/opencosmo_light.png
   :width: 600
   :alt: OpenCosmo Logo

|


Welcome to the OpenCosmo Toolkit Documentation
==============================================

The OpenCosmo Python Toolkit provides utilities for reading, writing and manipulating data from cosmological simulations produced by the Cosmolgical Physics and Advanced Computing (CPAC) group at Argonne National Laboratory. It can be used to work with smaller quantities data retrieved with the CosmoExplorer, as well as the much larget datasets these queries draw from. The OpenCosmo toolkit integrates with standard tools such as AstroPy, and allows you to manipulate data in a fully-consistent cosmological context.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   first_steps


.. toctree::
   :maxdepth: 2
   :caption: General Usage

   io
   main_api
   cols
   collections
   units

.. toctree::
   :maxdepth: 1
   :caption: Advanced Usage

   mpi
   performance

.. toctree::
   :maxdepth: 1
   :caption: Analysis and Visualization

   analysis

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   io_ref
   dataset_ref
   collection_ref
   parameters_ref
   spatial_ref
   analysis_ref

.. toctree::
   :maxdepth: 1
   :caption: Changelog
   :glob:

   changelog/*.*.*



