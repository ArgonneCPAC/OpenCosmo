from opencosmo.parameters import hacc


def get_origin_parameters(origin: str):
    if origin == "HACC":
        return hacc.ORIGIN_PARAMETERS
    else:
        raise ValueError(f"Unknown dataset origin {origin}")
