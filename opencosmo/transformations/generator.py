from typing import Protocol

from h5py import Dataset, Group

from opencosmo.file import broadcast_read
from opencosmo.transformations import transformation as t


class TransformationGenerator(Protocol):
    """
    A transformation generator is a callable that returns a transformation
    or set of transformations based on information stored in the attributes of a given
    dataset. Examples include units stored as attributes
    """

    def __call__(self, input: Dataset) -> t.TransformationDict: ...


@broadcast_read
def generate_transformations(
    input: Group,
    generators: list[TransformationGenerator],
    existing: t.TransformationDict = {},
) -> t.TransformationDict:
    """
    Generate transformations based on the input dataset and a list of generators.
    Generated transformations will always be run before other transformations.

    The logic is that generators rely on data that will not be accessible after
    the data is moved into memory, and so in some sense "precede" transformations
    that only operate on the in-memory representation.
    """
    for dataset in input.values():
        for gen in generators:
            generated_transformations = gen(dataset)
            if generated_transformations is not None:
                for (
                    transformation_type,
                    transformations,
                ) in generated_transformations.items():
                    existing_transformation = existing.get(transformation_type, [])
                    existing[transformation_type] = (
                        transformations + existing_transformation
                    )
    return existing
