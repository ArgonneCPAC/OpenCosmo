Parameters and the OpenCosmoHeader
==================================

Every OpenCosmo files comes with a set of parameters that describe the dataset it was draw from and other relevant information. These parameters are parsed into an :py:class:`OpenCosmoHeader <opencosmo.header.OpenCosmoHeader>` object which can be used throughout the code to verify properties of the dataset, or for reference.

Datasets and collections will allways make their header avaliable with the :py:attr:`.header <opencosmo.Dataset.header>` attribute. You can acccess the parameters in a given header with the :py:attr:`.parameters <opencosmo.header.OpenCosmoHeader.parameters>` attribute. The actual parameters available will vary depending on the type of data in the file.

.. autoclass:: opencosmo.header.OpenCosmoHeader
   :members:
   :undoc-members:
   :exclude-members: with_region, write
   :member-order: bysource

Available Metadata
-------------------

Every top-level key returned by :py:attr:`header.parameters <opencosmo.header.OpenCosmoHeader.parameters>` is also directly accessible as an attribute on the header itself, and on any :py:class:`Dataset <opencosmo.Dataset>` or collection built from it: ``header.simulation`` and ``header.parameters["simulation"]`` are equivalent, as are ``dataset.simulation`` and ``dataset.header.simulation``.

Because the set of available keys depends on the origin and data type of a given file, it will change depending on the specific dataset that is open. The table below is generated directly from the parameter models registered in ``opencosmo.dtypes``, so it always reflects the same access paths the library resolves at runtime. To see exactly what is available for a specific dataset, call :py:attr:`header.parameters <opencosmo.header.OpenCosmoHeader.parameters>` or use ``dir()``/tab-completion on the header or dataset.

.. metadata-table::

Cosmology
---------

Most OpenCosmo files will contain cosmology parameters, which describe the cosmology the simulation was run under. In general you will not interact with this parameter block directly. Instead, requiresting it will return an astropy.cosmology.Cosmology object. Dataset and collections will generally make this object available directly with the ``.cosmology`` attribute.

.. autoclass:: opencosmo.dtypes.cosmology.CosmologyParameters
   :members:
   :undoc-members:
   :exclude-members: model_config, ACCESS_PATH, ACCESS_TRANSFORMATION
   :member-order: bysource

Simulation Parameters
---------------------

Data that was originally produced by HACC will contain the parameters that were used to initialize the simulation. Datasets and collections will generally make these paramters available with the ``.simulation`` attribute.

.. autoclass:: opencosmo.dtypes.hacc.HaccSimulationParameters
   :members:
   :undoc-members:
   :exclude-members: model_config,empty_string_to_none,cosmology_parameters,ACCESS_PATH
   :member-order: bysource


.. autoclass:: opencosmo.dtypes.hacc.HaccHydroSimulationParameters
   :members:
   :undoc-members:
   :exclude-members: model_config
   :member-order: bysource

