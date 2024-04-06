"""Top-level sub-package for MPNN."""

__author__ = """Star Liu"""
__email__ = "StarLiu@seas.upenn.edu"
__version__ = "0.0.1"

from . import mpnn

__all__ = [
    "mpnn",
]

for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
