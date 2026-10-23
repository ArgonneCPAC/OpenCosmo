``SimulationCollection`` now accepts lightcones as members, and using a collection of
structure collections as a context manager no longer raises an ``AttributeError``.
``SimulationCollection.match`` also keeps the collection's key order instead of moving
the matched dataset to the front.
