Opening a file whose header has no cosmology block, or more than one, now raises a ``ValueError`` instead of a bare ``IndexError`` or silently using the first block.
