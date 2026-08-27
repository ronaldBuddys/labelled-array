"""Carrying values along a dimension: the forward fill.

The fill is over the *mask*, not over NaN, in the same way everything else in
this package is: absence is what the mask records, and a NaN that is masked in
is a real observation of nothing. A caller who wants NaN treated as absence can
say so by building the mask that way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["ffill"]


def _positions_along(axis: int, shape: tuple[int, ...]) -> np.ndarray:
    """Return each cell's own position along ``axis``, broadcast to ``shape``."""
    spread = [1] * len(shape)
    spread[axis] = shape[axis]
    return np.broadcast_to(np.arange(shape[axis]).reshape(spread), shape)


def ffill(array: Any, dim: str, limit: int | None = None) -> Any:
    """Carry the last present position forward along ``dim``.

    See :meth:`~labelled_array.LabelledArray.ffill`.
    """
    array.dims._require([dim])
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if array.mask is None:
        # Every cell is populated, so there is nothing to carry and no reason
        # to pay for a gather over the whole array.
        return array._replace()

    axis = array.dims.axis(dim)
    present = array.mask
    own = _positions_along(axis, array.shape)
    # The running maximum of "my position if I am present, else -1" gives every
    # cell the position of the last present cell at or before it, or -1 when
    # nothing precedes it.
    last = np.maximum.accumulate(np.where(present, own, -1), axis=axis)
    carry = ~present & (last >= 0)
    if limit is not None:
        carry &= (own - last) <= limit

    # A cell that is not carrying gathers from itself, which keeps its own fill
    # value and its own mask entry, and keeps -1 away from the backend: torch
    # rejects a negative gather index and JAX would clamp it.
    source = np.where(carry, last, own)
    cells = array._map_cells(
        lambda backend, values: backend.take_along_axis(values, source, axis)
    )
    return array._replace(**cells)
