from typing import Any

import numpy as np


def insert(
    storage: dict[str, np.ndarray], index: int, values_to_insert: dict[str, Any]
):
    if isinstance(values_to_insert, dict):
        for name, value in values_to_insert.items():
            storage[name][index] = value
        return storage

    name = next(iter(storage.keys()))
    storage[name][index] = values_to_insert
