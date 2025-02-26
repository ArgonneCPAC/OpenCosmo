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
            generated_transformations = generator(dataset)
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
