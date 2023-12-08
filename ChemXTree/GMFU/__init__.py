"""Top-level package for GMF Unit."""

__author__ = """Star Liu"""
__email__ = "StarLiu@seas.upenn.edu"
__version__ = "0.0.1"

from . import model, utils
from .tabular_data_modules import TabularDatamodule
from .tabular_model import TabularModel

__all__ = [
    "model", # GMFU/model
    "utils", #.py
    "TabularDatamodule", # GMFU/tabular_data_modules
    "TabularModel", #tabular_model.py
]

for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
