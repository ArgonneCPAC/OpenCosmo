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


def test_cache_passthrough():
    data = {}
    for name in "abcdefg":
        data[name] = np.random.randint(0, 1000, 10_000)
    cache = ColumnCache.empty()
    cache.add_data(data)

    indices2 = np.sort(np.random.choice(10_000, 1000, replace=False))
    index2 = SimpleIndex(indices2)
    cache2 = cache.take(index2)

    indices3 = np.sort(np.random.choice(1000, 100, replace=False))
    index3 = SimpleIndex(indices3)
    cache3 = cache2.take(index3)

    cached_data = cache3.get_columns("abcdefg")
    for colname, column in cached_data.items():
        assert np.all(column == data[colname][indices2[indices3]])
    assert set(cache3._ColumnCache__columns.keys()) == set("abcdefg")
    assert len(cache2._ColumnCache__columns) == 0


def test_cache_passthrough_delete():
    data = {}
    for name in "abcdefg":
        data[name] = np.random.randint(0, 1000, 10_000)
    cache = ColumnCache.empty()
    cache.add_data(data)

    indices2 = np.sort(np.random.choice(10_000, 1000, replace=False))
    index2 = SimpleIndex(indices2)
    cache2 = cache.take(index2)

    indices3 = np.sort(np.random.choice(1000, 100, replace=False))
    index3 = SimpleIndex(indices3)
    cache3 = cache2.take(index3)

    del cache2

    cached_data = cache3.get_columns("abcdefg")
    for colname, column in cached_data.items():
        assert np.all(column == data[colname][indices2[indices3]])
    assert len(cache3._ColumnCache__columns.keys()) == 0
