"""Exception hierarchy for :mod:`labelled_array`."""

__all__ = [
    "LabelledArrayError",
    "DimensionError",
    "LabelNotFoundError",
    "BackendError",
    "BackendDtypeError",
    "AdapterError",
    "DuplicateIndexError",
]


class LabelledArrayError(Exception):
    """Base class for every error raised by this package."""


class DimensionError(LabelledArrayError):
    """A dimension name, label set or shape is invalid or inconsistent."""


class LabelNotFoundError(KeyError, LabelledArrayError):
    """Requested labels are absent from a dimension.

    Subclasses :class:`KeyError` so that label lookup failures can be caught
    with the same idiom used for mappings.
    """

    def __init__(self, dim: str, missing) -> None:
        self.dim = dim
        self.missing = list(missing)
        super().__init__(f"labels not found in dimension {dim!r}: {self.missing}")

    def __str__(self) -> str:  # KeyError repr-quotes its argument; undo that.
        return self.args[0]


class BackendError(LabelledArrayError):
    """An array backend is unavailable, unknown, or cannot handle an object."""


class BackendDtypeError(BackendError):
    """A backend cannot represent a dtype (e.g. strings on torch)."""


class AdapterError(LabelledArrayError):
    """A DataFrame adapter is unavailable, unknown, or cannot handle an object."""


class DuplicateIndexError(LabelledArrayError):
    """DataFrame rows are not unique over the dimension columns.

    Multiple rows would map to the same array cell, so the conversion is
    ambiguous and is refused.
    """
