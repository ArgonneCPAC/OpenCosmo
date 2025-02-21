import h5py
from pydantic import BaseModel


def read_header_attributes(
    file: h5py.File, header_path: str, parameter_model: BaseModel, **kwargs
):
    header_data = file["header"][header_path].attrs
    parameters = parameter_model(**header_data, **kwargs)
    return parameters
