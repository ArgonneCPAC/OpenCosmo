Simulation Parameters
=====================

You can access the parameters of the simulation any dataset is drawn from with :py:attr:`opencosmo.Dataset.simulation`. All datasets regardless of simulation will have the parameters in :py:class:`opencosmo.parameters.SimulationParameters`. Hydrodynamic simulations will additionally contain the parameters in :py:class:`opencosmo.parameters.HydroSimulationParameters`


.. autoclass:: opencosmo.parameters.SimulationParameters
   :members:
   :undoc-members:
   :exclude-members: model_config,empty_string_to_none,cosmology_parameters
   :member-order: bysource


.. autoclass:: opencosmo.parameters.HydroSimulationParameters
   :members:
   :undoc-members:
   :exclude-members: model_config
   :member-order: bysource

.. autoclass:: opencosmo.parameters.SubgridParameters
   :members:
   :undoc-members:
   :exclude-members: model_config
   :member-order: bysource
