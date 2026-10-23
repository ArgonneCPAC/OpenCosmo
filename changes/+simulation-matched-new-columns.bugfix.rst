``SimulationCollection.with_new_columns`` now realigns a matched collection before adding
columns, so per-dataset arrays sized to the matched rows are accepted after operations
such as ``filter`` or ``take`` on the active source.
