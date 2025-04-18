Collections
===========

Collections within a single file can always be loaded with :py:func:`opencosmo.open`. Collections can be treated like read-only dictionaries. Dataset names can be retrieved with :py:meth:`keys`, the datasets can be accessed with :py:meth:`values` or :code:`Collection[key]`, and iteration can be done with :py:meth:`items`.


.. autoclass:: opencosmo.SimulationCollection
   :members:
   :exclude-members: open,read,close,write
   :undoc-members:
   :member-order: bysource
   

.. autoclass:: opencosmo.StructureCollection
   :members:
   :exclude-members: open,read,close,write
   :undoc-members:
   :member-order: bysource
