import numpy as np

from opencosmo.column.cache import ColumnCache
from opencosmo.index import SimpleIndex


def test_cache_take():
    data = {}
    for name in "abcdefg":
        data[name] = np.random.randint(0, 1000, 10_000)
    cache = ColumnCache.empty()
    cache.register_column_group(0, set("abcdefg"))
    cache.add_data(data)

    index2 = np.sort(np.random.choice(10_000, 100, replace=False))

    cache2 = cache.take(index2)
    cache2.register_column_group(0, set("abcdefg"))
    cached_data = cache2.get_columns("abcdefg")
    for colname, column in cached_data.items():
        assert np.all(column == data[colname][index2])


def test_cache_passthrough():
    data = {}
    for name in "abcdefg":
        data[name] = np.random.randint(0, 1000, 10_000)
    cache = ColumnCache.empty()
    cache.register_column_group(0, set("abcdefg"))
    cache.add_data(data)

    index2 = np.sort(np.random.choice(10_000, 1000, replace=False))
    cache2 = cache.take(index2)
    cache2.register_column_group(0, set("abcdefg"))

    index3 = np.sort(np.random.choice(1000, 100, replace=False))
    cache3 = cache2.take(index3)
    cache3.register_column_group(0, set("abcdefg"))

    cached_data = cache3.get_columns("abcdefg")
    for colname, column in cached_data.items():
        assert np.all(column == data[colname][index2[index3]])
    assert set(cache3.columns) == set("abcdefg")
    assert len(cache2.columns) == 0


def test_cache_passthrough_delete():
    data = {}
    for name in "abcdefg":
        data[name] = np.random.randint(0, 1000, 10_000)
    cache = ColumnCache.empty()
    cache.register_column_group(0, set("abcdefg"))
    cache.add_data(data)

    index2 = np.sort(np.random.choice(10_000, 1000, replace=False))
    cache2 = cache.take(index2)
    cache2.register_column_group(0, set("abcdefg"))
    _ = cache2.get_columns("abcdefg")

    index3 = np.sort(np.random.choice(1000, 100, replace=False))
    cache3 = cache2.take(index3)
    cache3.register_column_group(0, set("abcdefg"))

    assert set(cache3.columns) == set()
    del cache2
    assert set(cache3.columns) == set("abcdefg")
    cached_data = cache3.get_columns("abcdefg")

    for colname, column in cached_data.items():
        assert np.all(column == data[colname][index2[index3]])


def test_cache_passthrough_twice_delete():
    data = {}
    for name in "abcdefg":
        data[name] = np.random.randint(0, 1000, 10_000)
    cache = ColumnCache.empty()
    cache.register_column_group(0, set("abcdefg"))
    cache.add_data(data)

    index2 = np.sort(np.random.choice(10_000, 1000, replace=False))
    cache2 = cache.take(index2)
    cache2.register_column_group(0, set("abcdefg"))

    index3 = np.sort(np.random.choice(1000, 100, replace=False))
    cache3 = cache2.take(index3)
    cache3.register_column_group(0, set("abcdefg"))

    assert set(cache2.columns) == set()
    del cache
    assert set(cache2.columns) == set("abcdefg")

    cached_data = cache3.get_columns("abcdefg")

    for colname, column in cached_data.items():
        assert np.all(column == data[colname][index2[index3]])
    assert set(cache3.columns) == set("abcdefg")
