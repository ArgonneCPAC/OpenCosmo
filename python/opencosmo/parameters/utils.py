def empty_string_to_none(v):
    if isinstance(v, str) and v == "":
        return None
    return v
