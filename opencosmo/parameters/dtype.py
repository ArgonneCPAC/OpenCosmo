from opencosmo.parameters import hacc


def get_dtype_parameters(origin: str, dtype: str):
    if origin == "HACC":
        dtype_params = hacc.DATATYPE_PARAMETERS
    else:
        raise ValueError(f"Unknown dataset origin {origin}")
    return dtype_params[dtype]
