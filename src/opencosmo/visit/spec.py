from typing import NewType

"""
A visitor is a function that computes a quantity or quantities for some dataset or
collection based on some set subset of the quantities in that collection.

It is distinct from a derived column because it can be used for complex computations
that involve more than just algebraic combinations of columns that already exist. For
example, you may have a StructureCollection of halos, and want to compute some summary
statistic that will be included as part of the halo properties, but uses the halo
particles.

The visitor spec defines how we know which pieces of data are necessary to do the
computation. This allows us to only request the data we actually need. Different objects
may define different specs.
"""

# Datasets visitors can request columns
DatasetVisitorSpec = NewType("DatasetVisitorSpec", list[str])

# Structure collection visitors can request columns from all of their datasets
StructureCollectionVisitorSpec = NewType(
    "StructureCollectionVisitorSpec", dict[str, list[str]]
)

LightconeVisitorSpec = DatasetVisitorSpec
