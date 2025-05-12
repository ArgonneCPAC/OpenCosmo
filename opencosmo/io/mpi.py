from mpi4py import MPI
import numpy as np
from .schemas import FileSchema, DatasetSchema, SimCollectionSchema, StructCollectionSchema, ColumnSchema, IdxLinkSchema, StartSizeLinkSchema
from .protocols import DataSchema
from copy import copy

def verify_structure(schemas: dict[str, DataSchema], comm: MPI.Comm):
    verify_names(schemas, comm)
    verify_types(schemas, comm)


def verify_names(schemas: dict[str, DataSchema], comm: MPI.Comm):
    names = set(schemas.keys())
    all_names = comm.allgather(names)
    if not all(ns == all_names[0] for ns in all_names[1:]):
        raise ValueError("Tried to combine a collection of schemas with different names!")

def verify_types(schemas: dict[str, DataSchema], comm: MPI.Comm):
    types = list(str(type(c)) for c in schemas.values())
    types.sort()
    all_types = comm.allgather(types)
    if not all(ts == all_types[0] for ts in all_types[1:]):
        raise ValueError("Tried to combine a collection of schemas with different types!")


def combine_file_schemas(schema: FileSchema, comm: MPI.Comm = MPI.COMM_WORLD):
    verify_structure(schema.children, comm)

    if comm.Get_rank() == 0:
        new_schema = FileSchema()
    else:
        new_schema = None

    for child_name in schema.children:
        child_name_ = comm.bcast(child_name)
        child = schema.children[child_name_]
        new_child = combine_file_child(child, comm)
        if comm.Get_rank() == 0:
            new_schema.add_child(new_child, child_name)
    return new_schema

def combine_file_child(schema: DataSchema, comm: MPI.Comm):
    match schema:
        case DatasetSchema():
            return combine_dataset_schemas(schema, comm)
        case SimCollectionSchema():
            return combine_simcollection_schema(schema, comm)
        case StructCollectionSchema():
            return combine_structcollection_schema(schema, comm)


def combine_dataset_schemas(schema: DatasetSchema, comm: MPI.Comm):
    rank = comm.Get_rank()
    verify_structure(schema.columns, comm)
    verify_structure(schema.links, comm)

    if rank == 0:
        new_schema = DatasetSchema(schema.header)
    else:
        new_schema = None

    for colname in schema.columns.keys():
        colname_ = comm.bcast(colname)
        new_column = combine_column_schemas(schema.columns[colname_], comm)
        if rank == 0:
            new_schema.add_child(new_column, colname)


    if len(schema.links) > 0:
        new_links = combine_links(schema.links, comm)
        if rank == 0:
            for name, link in new_links.items():
                new_schema.add_child(link, name)
    return new_schema

def combine_links(links: dict[str, StartSizeLinkSchema | IdxLinkSchema], comm: MPI.Comm):
    new_links = {}
    for name, link in links.items():
        if isinstance(link, StartSizeLinkSchema):
            new_links[name] = combine_start_size_link_schema(link, comm)
        else:
            new_links[name] = combine_idx_link_schema(link, comm)

    return new_links

def combine_idx_link_schema(schema: IdxLinkSchema, comm: MPI.Comm):
    column_schema = combine_column_schemas(schema.column, comm)
    new_schema = copy(schema)
    new_schema.column = column_schema
    return new_schema

def combine_start_size_link_schema(schema: StartSizeLinkSchema, comm: MPI.Comm):
    start_column_schema = combine_column_schemas(schema.start, comm)
    size_column_schema = combine_column_schemas(schema.size, comm)
    new_schema = copy(schema)
    new_schema.start = start_column_schema
    new_schema.size = size_column_schema
    return new_schema

def combine_simcollection_schema(schema: SimCollectionSchema, comm: MPI.Comm):
    rank = comm.Get_rank()
    verify_structure(schema.children, comm) 

    child_names = schema.children.keys()

    if rank == 0:
        new_schema = SimCollectionSchema()

    else:
        new_schema = None

    for child_name in child_names:
        child_name_ = comm.bcast(child_name)
        child = schema.children[child_name_]
        match child:
            case StructCollectionSchema():
                new_child = combine_structcollection_schema(child, comm)
            case DatasetSchema():
                new_child = combine_dataset_schemas(schema, comm)
        if rank == 0:
            new_schema.add_child(new_child, child_name)
    return new_schema




def combine_structcollection_schema(schema: StructCollectionSchema, comm: MPI.Comm):
    rank = comm.Get_rank()
    child_names = set(schema.children.keys())
    all_child_names = comm.allgather(child_names)
    if not all(cns == all_child_names[0] for cns in all_child_names[1:]):
        raise ValueError("Tried to combine ismulation collections with different children!")

    child_types = set(str(type(c)) for c in schema.children.values())
    all_child_types = comm.allgather(child_types)
    if not all(cts == all_child_types[0] for cts in all_child_types[1:]):
        raise ValueError("Tried to combine ismulation collections with different children!")

    new_schema = StructCollectionSchema(schema.header) if rank == 0 else None
    child_names = list(child_names)
    child_names.sort()


    for i, name in enumerate(child_names):
        cn = comm.bcast(name)
        child = schema.children[cn]
        match child:
            case DatasetSchema():
            
                new_child = combine_dataset_schemas(child, comm)
        if rank == 0:
            new_schema.add_child(new_child, cn)

    return new_schema




   
    

def combine_column_schemas(schema: ColumnSchema, comm: MPI.Comm):
    rank = comm.Get_rank()
    lengths = comm.allgather(len(schema.index))
    rank_offsets = np.insert(np.cumsum(lengths), 0, 0)[:-1]
    rank_offset = rank_offsets[rank]
    schema.set_offset(rank_offset)

    indices = comm.allgather(schema.index)
    new_index = indices[0].concatenate(*indices[1:])

    return ColumnSchema(schema.name, new_index, schema.source)




