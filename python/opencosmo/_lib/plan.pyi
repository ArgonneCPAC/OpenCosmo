from opencosmo.index import IndexArray

def get_chunked_read_plan(
    start: IndexArray,
    size: IndexArray,
    hdf5_chunk_size: int | None = None,
    max_buffer_rows: int | None = None,
) -> tuple[IndexArray, IndexArray, IndexArray, IndexArray, IndexArray, IndexArray]: ...
def get_simple_read_plan(
    index: IndexArray,
    hdf5_chunk_size: int | None = None,
    max_buffer_rows: int | None = None,
) -> tuple[IndexArray, IndexArray, IndexArray, IndexArray, IndexArray, IndexArray]: ...
