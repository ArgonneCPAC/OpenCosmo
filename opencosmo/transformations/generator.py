from typing import Protocol

from h5py import Dataset, Group

from opencosmo.transformations import transformation as t


class TransformationGenerator(Protocol):
    """
    A transformation generator is a callable that returns a transformation
    or set of transformations based on information stored in the attributes of a given
    dataset. Examples include units stored as attributes
    """

    def __call__(self, input: Dataset) -> dict[str, list[t.Transformation]]: ...


def generate_transformations(
    input: Group,
    generators: list[TransformationGenerator],
    existing: dict[str, list[t.Transformation]] = {},
) -> dict[str, list[t.Transformation]]:
    """
    Generate transformations based on the input dataset and a list of generators
    """
    for dataset in input.values():
        for generator in generators:
            transformations = generator(dataset)
            if transformations is not None:
                for transformation_type, transformation in transformations.items():
                    if transformation_type not in existing:
                        existing[transformation_type] = []
                    existing[transformation_type].extend(transformation)
    return existing
