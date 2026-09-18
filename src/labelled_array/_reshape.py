"""Engine-agnostic conversion between flat rows and dense labelled arrays.

Both DataFrame adapters hand numpy arrays to the functions here, so pandas and
polars produce identical results by construction rather than by coincidence.
:func:`factorize` is also the reference an adapter's native factorization has
to reproduce exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from labelled_array._utils import duplicates
from labelled_array.dimensions import Dimensions
from labelled_array.exceptions import DuplicateIndexError

__all__ = [
    "default_fill_value",
    "factorize",
    "flat_positions",
    "natural_index",
    "natural_rank",
    "scatter",
    "take_flat",
]


def default_fill_value(dtype: Any) -> Any:
    """Return the value used for cells no row populates.

    Missing cells are recorded by the presence mask, so the fill is never read
    as data by this package. It is chosen to be unmistakable to anything that
    reads the array *without* the mask -- a bare ``array["x"]``, another
    library, or an array built with ``mask=False``:

    ==========================  ===========================================
    dtype                       fill
    ==========================  ===========================================
    float, complex              ``nan``
    datetime64, timedelta64     ``NaT``, in the dtype's own unit
    signed integer              the dtype's minimum, e.g. ``-2**63``
    unsigned integer            the dtype's maximum, e.g. ``2**64 - 1``
    object                      ``None``
    str, bytes                  the empty string
    bool                        ``False`` -- a bool has no value to spare
    ==========================  ===========================================

    Pass ``fill_value=`` wherever cells are invented to choose another.

    Parameters
    ----------
    dtype : numpy dtype
        Dtype of the array being filled.

    Returns
    -------
    scalar
        The fill for ``dtype``.

    Examples
    --------
    >>> default_fill_value("int32")
    -2147483648
    >>> default_fill_value("uint8")
    255
    """
    kind = np.dtype(dtype).kind
    if kind in "fc":
        return np.nan
    if kind in "Mm":
        # A unit-less timedelta64("NaT") is deprecated by numpy; take the unit
        # of the array being filled so the sentinel matches it exactly.
        unit = np.datetime_data(np.dtype(dtype))[0]
        return (
            np.datetime64("NaT", unit) if kind == "M" else np.timedelta64("NaT", unit)
        )
    if kind in "iu":
        # An integer has no null, and a zero would pass for data. The extreme
        # of the range cannot: it is what numpy itself uses for NaT, and a sum
        # that includes one is wrong by enough to be noticed.
        limits = np.iinfo(dtype)
        return int(limits.min if kind == "i" else limits.max)
    if kind == "b":
        return False
    if kind in "US":
        return ""
    if kind == "O":
        return None
    return 0


def factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unique labels of ``values`` and each entry's position in them.

    Labels are sorted where the dtype allows it, which keeps the array layout
    independent of row order. Object columns holding values that cannot be
    compared fall back to order of first appearance.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional column of dimension labels, one entry per row.

    Returns
    -------
    labels : numpy.ndarray
        The unique values.
    codes : numpy.ndarray
        Position of each entry within ``labels``.
    """
    if values.dtype.kind != "O":
        return np.unique(values, return_inverse=True)

    # Object columns -- what every string column becomes -- would make
    # np.unique sort all n rows with Python-level comparisons. Hashing the rows
    # and sorting only the distinct values is an order of magnitude cheaper.
    seen: dict[Any, int] = {}
    codes = np.empty(len(values), dtype=np.intp)
    for row, value in enumerate(values.tolist()):
        codes[row] = seen.setdefault(value, len(seen))
    labels = np.empty(len(seen), dtype=object)
    for value, position in seen.items():
        labels[position] = value

    try:
        order = np.argsort(labels, kind="stable")
    except TypeError:
        return labels, codes  # mixed types cannot be sorted; keep first-seen order
    ranks = np.empty(order.size, dtype=np.intp)
    ranks[order] = np.arange(order.size)
    return labels[order], ranks[codes]


def flat_positions(
    codes: list[np.ndarray], dims: Dimensions, n_rows: int
) -> np.ndarray:
    """Return the flat array position of each row, checking they are unique.

    Parameters
    ----------
    codes : list of numpy.ndarray
        Per-dimension position of each row.
    dims : Dimensions
        Dimensions of the array being filled.
    n_rows : int
        Number of rows, needed when there are no dimensions at all.

    Raises
    ------
    DuplicateIndexError
        If two rows map to the same cell, which would make the conversion
        ambiguous. The message names the labels of the offending cells.
    """
    shape, names = dims.shape, list(dims.names)
    if not shape:
        if n_rows > 1:
            raise DuplicateIndexError(
                f"{n_rows} rows map to a zero-dimensional array, which holds one cell"
            )
        return np.zeros(n_rows, dtype=np.intp)

    positions = np.ravel_multi_index(tuple(codes), shape)
    duplicated = duplicates(positions)
    if not duplicated:
        return positions

    cells = np.unravel_index(duplicated[:5], shape)
    labels = {name: dims[name].tolist() for name in names}
    examples = [
        {
            name: labels[name][axis[row]]
            for name, axis in zip(names, cells, strict=False)
        }
        for row in range(min(len(duplicated), 5))
    ]
    raise DuplicateIndexError(
        f"{len(duplicated)} cell(s) are populated by more than one row; "
        f"dimensions {names} must identify rows uniquely. "
        f"First offending cells: {examples}"
    )


def scatter(
    values: np.ndarray,
    positions: np.ndarray,
    shape: tuple[int, ...],
    fill_value: Any = None,
    dtype: Any = None,
) -> np.ndarray:
    """Return a dense array of ``shape`` holding ``values`` at ``positions``.

    Parameters
    ----------
    values : numpy.ndarray
        One value per row.
    positions : numpy.ndarray
        Flat destination position of each row.
    shape : tuple of int
        Shape of the array to build.
    fill_value : scalar, optional
        Value for cells no row populates; defaults to
        :func:`default_fill_value` for the dtype.
    dtype : numpy dtype, optional
        Dtype of the result; defaults to the dtype of ``values``.
    """
    dtype = np.dtype(values.dtype if dtype is None else dtype)
    if fill_value is None:
        fill_value = default_fill_value(dtype)
    out = np.full(shape, fill_value, dtype=dtype)
    out.reshape(-1)[positions] = values
    return out


def take_flat(array: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return the entries of ``array`` at flat ``positions``."""
    return array.reshape(-1)[positions]


def natural_index(codes: list[np.ndarray]) -> np.ndarray:
    """Return a dense index counting rows within each preceding-code group.

    Rows are numbered ``0, 1, 2, ...`` within each distinct combination of the
    preceding dimensions' codes, in ascending order of the dimension being
    replaced. This is what collapses a dimension whose labels are unique per
    group (typically dates) from "every label seen anywhere" down to "the most
    labels any one group has".

    The labels of the dimension being replaced **must be unique within each
    group**. That is a correctness precondition, not a performance one: this
    counts rows, not distinct labels, so a label repeated within a group gets
    two positions and comes back as two rungs of the ladder -- finite,
    plausible and wrong. ``from_dataframe`` checks it and raises
    :class:`~labelled_array.DuplicateIndexError`; this function does not.
    :func:`natural_rank` is the index for labels that are meant to repeat.

    Parameters
    ----------
    codes : list of numpy.ndarray
        Codes of the preceding dimensions, then the codes of the dimension
        being replaced, as the last entry.

    Returns
    -------
    numpy.ndarray
        The within-group position of each row.
    """
    *preceding, target = codes
    order = np.lexsort([target, *reversed(preceding)])
    index = np.empty(len(target), dtype=np.intp)
    if not preceding:
        index[order] = np.arange(len(target))
        return index

    grouped = np.stack([code[order] for code in preceding])
    starts_group = np.empty(len(target), dtype=bool)
    starts_group[0] = True
    starts_group[1:] = np.any(grouped[:, 1:] != grouped[:, :-1], axis=0)
    group_start = np.maximum.accumulate(
        np.where(starts_group, np.arange(len(target)), 0)
    )
    index[order] = np.arange(len(target)) - group_start
    return index


def natural_rank(codes: list[np.ndarray]) -> np.ndarray:
    """Return a dense index ranking distinct labels within each preceding-code group.

    Where :func:`natural_index` counts rows, this counts *distinct labels*: rows
    sharing a label within a group share a position. That is the index a ladder
    shared by the rows beneath it needs -- the expiries of a long option chain,
    which repeat once per strike -- and it equals :func:`natural_index` wherever
    the labels are unique within each group.

    Parameters
    ----------
    codes : list of numpy.ndarray
        Codes of the preceding dimensions, then the codes of the dimension
        being replaced, as the last entry.

    Returns
    -------
    numpy.ndarray
        The within-group dense rank (0-based, ascending by label) of each row.
    """
    *preceding, target = codes
    n_rows = len(target)
    rank = np.empty(n_rows, dtype=np.intp)
    if n_rows == 0:
        return rank

    order = np.lexsort([target, *reversed(preceding)])
    ordered = target[order]
    starts_group = np.zeros(n_rows, dtype=bool)
    starts_group[0] = True
    for code in preceding:
        grouped = code[order]
        starts_group[1:] |= grouped[1:] != grouped[:-1]

    new_label = starts_group.copy()
    new_label[1:] |= ordered[1:] != ordered[:-1]
    # Distinct labels seen so far, less the count at the start of the row's
    # group, is the rank within the group.
    seen = np.cumsum(new_label) - 1
    at_group_start = np.maximum.accumulate(np.where(starts_group, seen, 0))
    rank[order] = seen - at_group_start
    return rank
