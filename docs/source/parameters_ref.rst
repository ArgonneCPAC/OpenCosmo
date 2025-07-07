Parameters and the OpenCosmoHeader
==================================

Every OpenCosmo files comes with a set of parameters that describe the dataset it was draw from and other relevant information. These parameters are parsed into an :py:class:`OpenCosmoHeader <opencosmo.header.OpenCosmoHeader>` object which can be used throughout the code to verify properties of the dataset, or for reference.

Datasets and collections will allways make their header avaliable with the :py:attr:`.header <opencosmo.Dataset.header>` attribute. You can acccess the parameters in a given header with the :py:attr:`.parameters <opencosmo.header.OpenCosmoHeader.parameters>` attribute. The actual parameters available will vary depending on the type of data in the file.

.. autoclass:: opencosmo.header.OpenCosmoHeader
   :members:
   :undoc-members:
   :exclude-members: with_region, write
   :member-order: bysource


Cosmology
---------

Most OpenCosmo files will contain cosmology parameters, which describe the cosmology the simulation was run under. In general you will not interact with this parameter block directly. Instead, requiresting it will return an astropy.cosmology.Cosmology object. Dataset and collections will generally make this object available directly with the :py:attr:`.cosmology <opencosmo.Dataset.cosmology>` attribute.

.. autoclass:: opencosmo.parameters.cosmology.CosmologyParameters
   :members:
   :undoc-members:
   :exclude-members: model_config, ACCESS_PATH, ACCESS_TRANSFORMATION
   :member-order: bysource

Simulation Parameters
---------------------

Data that was originally produced by HACC will contain the parameters that were used to initialize the simulation. Datasets and collections will generally make these paramters available with the :py:attr:`.simulation <opencosmo.Dataset.simulation>` attribute.

.. autoclass:: opencosmo.parameters.hacc.HaccSimulationParameters
   :members:
   :undoc-members:
   :exclude-members: model_config,empty_string_to_none,cosmology_parameters,ACCESS_PATH
   :member-order: bysource


.. autoclass:: opencosmo.parameters.hacc.HaccHydroSimulationParameters
   :members:
   :undoc-members:
   :exclude-members: model_config
   :member-order: bysource

