"""Arrays that carry their dimension names and labels.

A :class:`~labelled_array.core.LabelledArray` holds one or more equally shaped
arrays together with an ordered mapping of dimension name to unique labels,
aligns and reduces over them, and converts losslessly to and from pandas or
polars DataFrames.
"""

from importlib.metadata import version

from labelled_array._ops import GroupBy, Rolling
from labelled_array._reshape import default_fill_value
from labelled_array.core import LabelledArray
from labelled_array.dimensions import Dimensions
from labelled_array.exceptions import (
    AdapterError,
    BackendDtypeError,
    BackendError,
    DimensionError,
    DuplicateIndexError,
    LabelledArrayError,
    LabelNotFoundError,
)

# One source of truth: the version in pyproject.toml, as installed.
__version__ = version("labelled-array")

__all__ = [
    "AdapterError",
    "Dimensions",
    "GroupBy",
    "Rolling",
    "LabelledArray",
    "BackendDtypeError",
    "BackendError",
    "DimensionError",
    "DuplicateIndexError",
    "LabelNotFoundError",
    "LabelledArrayError",
    "__version__",
    "default_fill_value",
]
