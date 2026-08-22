"""Arrays that carry their dimension names and labels.

A :class:`~labelled_array.core.LabelledArray` holds one or more equally shaped
arrays together with an ordered mapping of dimension name to unique labels, and
converts losslessly to and from pandas or polars DataFrames.
"""

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

__version__ = "0.1.0"

__all__ = [
    "AdapterError",
    "Dimensions",
    "LabelledArray",
    "BackendDtypeError",
    "BackendError",
    "DimensionError",
    "DuplicateIndexError",
    "LabelNotFoundError",
    "LabelledArrayError",
    "__version__",
]
