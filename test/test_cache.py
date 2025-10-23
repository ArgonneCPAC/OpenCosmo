import numpy as np

from opencosmo.column.cache import ColumnCache
from opencosmo.index import SimpleIndex


def test_cache_take():
    data = {}
    for name in "abcdefg":
        data[name] = np.random.randint(0, 1000, 10_000)
    cache = ColumnCache.empty()
    cache.add_data(data)

    indices2 = np.sort(np.random.choice(10_000, 100, replace=False))
    index2 = SimpleIndex(indices2)

    cache2 = cache.take(index2)
    cached_data = cache2.get_columns("abcdefg")
    for colname, column in cached_data.items():
        assert np.all(column == data[colname][indices2])
