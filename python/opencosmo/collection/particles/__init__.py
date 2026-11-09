from importlib import import_module

from .particles import ParticleCollection

# register plugins
import_module(f"{__name__}.plugins")

__all__ = ["ParticleCollection"]
