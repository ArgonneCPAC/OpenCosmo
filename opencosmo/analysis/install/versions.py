import h5py

from opencosmo.analysis.install import get_specs


def get_file_versions(spec_name, file):
    file = h5py.File(file)
    specs = get_specs()
    spec = specs[spec_name]
    if spec.header_version_key is None:
        raise ValueError("This spec does not support file versions")
    if (
        "header" not in file.keys()
        or spec.header_version_key not in file["header"].keys()
    ):
        raise NotImplementedError

    versions = dict(file["header"][spec.header_version_key].attrs)
    return versions
