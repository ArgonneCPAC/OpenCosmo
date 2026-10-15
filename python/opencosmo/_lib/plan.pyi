from opencosmo.index import IndexArray

type Windows = tuple[
    IndexArray, IndexArray, IndexArray, IndexArray, IndexArray, IndexArray
]

def plan_windows_simple(
    index: IndexArray, n_rows: int, chunk_rows: int, max_chunks: int
) -> Windows | None: ...
def plan_windows_chunked(
    starts: IndexArray, sizes: IndexArray, n_rows: int, chunk_rows: int, max_chunks: int
) -> Windows | None: ...
