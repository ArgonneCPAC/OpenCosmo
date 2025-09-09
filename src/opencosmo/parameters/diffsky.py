from typing import ClassVar

from pydantic import BaseModel


class DiffskyVersionInfo(BaseModel, frozen=True):
    ACCESS_PATH: ClassVar[str] = "diffsky_versions"
    diffmah: str
    diffsky: str
    diffstar: str
    diffstarpop: str
    dsps: str
    jax: str
    numpy: str
