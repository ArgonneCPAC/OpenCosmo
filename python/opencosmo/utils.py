import keyword
import re


def normalize_kwarg_name(s: str) -> str:
    """
    Normalize a string into a valid Python identifier suitable for use
    as a function keyword argument name.
    """
    # Replace any run of characters that aren't letters, digits, or underscore with '_'
    s = re.sub(r"\W+", "_", s)

    # Strip leading/trailing underscores left over from replacement
    s = s.strip("_")

    # If it starts with a digit, prefix with an underscore
    if re.match(r"^\d", s):
        s = f"_{s}"

    # If it's empty after cleaning, fall back to a generic name
    if not s:
        s = "_"

    # If it collides with a Python keyword, append an underscore
    if keyword.iskeyword(s):
        s = f"{s}_"

    return s
