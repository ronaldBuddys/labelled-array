"""Small helpers with no dependencies on the rest of the package."""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "array_repr",
    "arrays_equal",
    "as_name_list",
    "duplicates",
    "is_null",
    "per_name",
]


def array_repr(arr: np.ndarray, full: bool = True) -> str:
    """Return ``arr`` as a call to ``numpy.array``.

    The dtype is always spelled out as a string, and floats are printed with
    every digit needed to reproduce them, so the result can be evaluated with
    only ``array`` (i.e. :func:`numpy.array`) in scope and gives back an
    identical array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to represent.
    full : bool, default True
        Print every element. When False, long arrays are summarised with an
        ellipsis and the result is no longer ``eval``-able.
    """
    with np.printoptions(
        threshold=sys.maxsize if full else 1000,
        floatmode="unique",
        nanstr="float('nan')",
        infstr="float('inf')",
    ):
        body = np.array2string(arr, separator=", ")
    return f"array({body}, dtype={arr.dtype.str!r})"


def is_null(array: np.ndarray) -> np.ndarray:
    """Return a boolean array marking null entries, per dtype.

    Only float, complex, datetime, timedelta and object arrays can hold a null;
    every other dtype reports no nulls at all.
    """
    kind = array.dtype.kind
    if kind in "fc":
        return np.isnan(array)
    if kind in "Mm":
        return np.isnat(array)
    if kind == "O":
        return (array != array) | np.equal(array, None)
    return np.zeros(array.shape, dtype=bool)


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Whether two arrays hold the same values, with nulls matching nulls.

    Unlike :func:`numpy.array_equal`, NaN, NaT and None compare equal to
    themselves, so arrays holding the same unpopulated cells are equal.
    """
    if left.shape != right.shape:
        return False
    if np.array_equal(left, right):
        return True
    left_null = is_null(left)
    if not np.array_equal(left_null, is_null(right)):
        return False
    return bool(np.array_equal(left[~left_null], right[~left_null]))


def duplicates(values: Sequence[Any] | np.ndarray) -> list:
    """Return the values appearing more than once.

    Sorting and comparing neighbours is far cheaper than counting in Python,
    and returns nothing at all on the common case of no duplicates. Values that
    cannot be sorted -- object arrays of mixed types -- fall back to counting,
    which reports them in order of first appearance.
    """
    array = values if isinstance(values, np.ndarray) else np.asarray(values)
    if array.size < 2:
        return []
    try:
        ordered = np.sort(array, kind="stable")
    except TypeError:
        counts = Counter(array.tolist())
        return [value for value, count in counts.items() if count > 1]
    repeated = ordered[1:][ordered[1:] == ordered[:-1]]
    return np.unique(repeated).tolist() if repeated.size else []


def as_name_list(names: str | Sequence[str] | None) -> list[str]:
    """Return ``names`` as a list, accepting a single name or None."""
    if names is None:
        return []
    return [names] if isinstance(names, str) else list(names)


def per_name(
    value: Any, names: Sequence[str], what: str, noun: str = "columns"
) -> dict:
    """Spread an option over ``names``, or validate a per-name mapping.

    Parameters
    ----------
    value : object or mapping of str to object
        One value for every name, or a value per named entry.
    names : sequence of str
        The names the option may be given for.
    what : str
        Name of the option, used in the error message.
    noun : str, default 'columns'
        What the names are, used in the error message.

    Returns:
    -------
    dict
        One entry per name, None where a mapping did not name it.
    """
    if not isinstance(value, Mapping):
        return dict.fromkeys(names, value)
    unknown = sorted(set(value) - set(names))
    if unknown:
        raise KeyError(f"{what} given for unknown {noun}: {unknown}")
    return {name: value.get(name) for name in names}
