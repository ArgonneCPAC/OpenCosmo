from .collection import StructureCollection
from .handler import LinkHandler, OomLinkHandler
from .io import open_linked_file, open_linked_files

__all__ = [
    "StructureCollection",
    "LinkHandler",
    "OomLinkHandler",
    "open_linked_files",
    "open_linked_file",
]
