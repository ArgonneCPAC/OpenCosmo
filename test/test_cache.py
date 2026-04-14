from uuid import uuid4

import numpy as np
from opencosmo.column.cache import ColumnCache


def _make_mapping(names):
    """Create a {name: UUID} mapping for testing."""
    return {name: uuid4() for name in names}


def _add_raw_data(cache, name_to_uuid, raw_data):
    """Add name-keyed data to cache via UUID-keyed interface."""
    uuid_data = {name_to_uuid[name]: {name: raw_data[name]} for name in raw_data}
    cache.add_data(uuid_data)


def _get_flat_data(cache, name_to_uuid):
    """Get data from cache, flattened to a plain name-keyed dict."""
    pairs = {(uuid, name) for name, uuid in name_to_uuid.items()}
    uuid_data = cache.get_data(pairs)
    return {name: arr for d in uuid_data.values() for name, arr in d.items()}


def test_cache_take():
    raw_data = {name: np.random.randint(0, 1000, 10_000) for name in "abcdefg"}
    name_to_uuid = _make_mapping("abcdefg")

    cache = ColumnCache.empty()
    cache.register_column_group(0, name_to_uuid)
    _add_raw_data(cache, name_to_uuid, raw_data)

    index2 = np.sort(np.random.choice(10_000, 100, replace=False))

    cache2 = cache.take(index2)
    cache2.register_column_group(0, name_to_uuid)
    cached_data = _get_flat_data(cache2, name_to_uuid)
    for colname, column in cached_data.items():
        assert np.all(column == raw_data[colname][index2])


def test_cache_passthrough():
    raw_data = {name: np.random.randint(0, 1000, 10_000) for name in "abcdefg"}
    name_to_uuid = _make_mapping("abcdefg")

    cache = ColumnCache.empty()
    cache.register_column_group(0, name_to_uuid)
    _add_raw_data(cache, name_to_uuid, raw_data)

    index2 = np.sort(np.random.choice(10_000, 1000, replace=False))
    cache2 = cache.take(index2)
    cache2.register_column_group(0, name_to_uuid)

    index3 = np.sort(np.random.choice(1000, 100, replace=False))
    cache3 = cache2.take(index3)
    cache3.register_column_group(0, name_to_uuid)

    cached_data = _get_flat_data(cache3, name_to_uuid)
    for colname, column in cached_data.items():
        assert np.all(column == raw_data[colname][index2[index3]])
    assert set(cache3.columns) == set("abcdefg")
    assert len(cache2.columns) == 0


def test_cache_passthrough_delete():
    raw_data = {name: np.random.randint(0, 1000, 10_000) for name in "abcdefg"}
    name_to_uuid = _make_mapping("abcdefg")

    cache = ColumnCache.empty()
    cache.register_column_group(0, name_to_uuid)
    _add_raw_data(cache, name_to_uuid, raw_data)

    index2 = np.sort(np.random.choice(10_000, 1000, replace=False))
    cache2 = cache.take(index2)
    cache2.register_column_group(0, name_to_uuid)
    _ = _get_flat_data(cache2, name_to_uuid)

    index3 = np.sort(np.random.choice(1000, 100, replace=False))
    cache3 = cache2.take(index3)
    cache3.register_column_group(0, name_to_uuid)

    assert set(cache3.columns) == set()
    del cache2
    assert set(cache3.columns) == set("abcdefg")
    cached_data = _get_flat_data(cache3, name_to_uuid)

    for colname, column in cached_data.items():
        assert np.all(column == raw_data[colname][index2[index3]])


def test_cache_passthrough_twice_delete():
    raw_data = {name: np.random.randint(0, 1000, 10_000) for name in "abcdefg"}
    name_to_uuid = _make_mapping("abcdefg")

    cache = ColumnCache.empty()
    cache.register_column_group(0, name_to_uuid)
    _add_raw_data(cache, name_to_uuid, raw_data)

    index2 = np.sort(np.random.choice(10_000, 1000, replace=False))
    cache2 = cache.take(index2)
    cache2.register_column_group(0, name_to_uuid)

    index3 = np.sort(np.random.choice(1000, 100, replace=False))
    cache3 = cache2.take(index3)
    cache3.register_column_group(0, name_to_uuid)

    assert set(cache2.columns) == set()
    del cache
    assert set(cache2.columns) == set("abcdefg")

    cached_data = _get_flat_data(cache3, name_to_uuid)

    for colname, column in cached_data.items():
        assert np.all(column == raw_data[colname][index2[index3]])
    assert set(cache3.columns) == set("abcdefg")
