"""Reductions over dimensions, grouped reductions, and rolling windows.

Every operation here is mask-aware in the same way: cells the mask marks absent
are replaced with the operation's identity before the backend reduces anything,
and the count of populated cells decides both the result's mask and, for a mean,
what it is divided by. Nothing here looks at NaN -- absence is what the mask
records, and a NaN that is masked in is data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from labelled_array._align import concat
from labelled_array._reshape import default_fill_value, factorize
from labelled_array._utils import arrays_equal, as_name_list
from labelled_array.dimensions import as_1d
from labelled_array.exceptions import DimensionError

__all__ = ["OPS", "GroupBy", "Rolling", "groupby", "reduce", "rolling"]

#: Reductions offered over dimensions, groups and rolling windows.
OPS = ("sum", "mean", "min", "max")

#: The backend reduction each of them is built from.
_BASE_OP = {"sum": "sum", "mean": "sum", "min": "min", "max": "max"}


def _check_op(op: str) -> str:
    """Return ``op``, or raise if it is not a reduction this package offers."""
    if op not in OPS:
        raise ValueError(f"op must be one of {OPS}, got {op!r}")
    return op


def identity(op: str, dtype: Any) -> Any:
    """Return the value an absent cell takes so that it cannot affect ``op``.

    Zero for a sum, and the largest or smallest representable value for a
    minimum or a maximum, so the reduction itself has nothing to skip.

    Raises:
    ------
    TypeError
        If the dtype has no such value: strings and objects cannot be reduced
        at all, and dates cannot be summed.
    """
    dtype = np.dtype(dtype)
    kind = dtype.kind
    if kind in "OUSV":
        raise TypeError(f"cannot take the {op} of a variable of dtype {dtype}")
    if _BASE_OP[op] == "sum":
        if kind == "M":
            raise TypeError(
                f"cannot take the {op} of a variable of dtype {dtype}: "
                "dates can be compared but not added"
            )
        return np.zeros((), dtype=dtype)[()]
    if kind == "c":
        raise TypeError(f"cannot take the {op} of a variable of dtype {dtype}")
    extreme = "max" if op == "min" else "min"
    if kind == "f":
        return np.inf if op == "min" else -np.inf
    if kind == "b":
        return op == "min"
    if kind in "iu":
        return getattr(np.iinfo(dtype), extreme)
    # datetime64 and timedelta64 are int64 underneath, and its minimum is NaT.
    bound = np.iinfo(np.int64).max if op == "min" else np.iinfo(np.int64).min + 1
    return np.array(bound).astype(dtype)[()]


def _counts(array: Any, axes: Sequence[int]) -> np.ndarray:
    """Return the number of populated cells behind each cell of the result."""
    if array.mask is None:
        spanned = 1
        for axis in axes:
            spanned *= array.shape[axis]
        return np.full(_reduced_shape(array.shape, axes), spanned, dtype=np.intp)
    return array.mask.sum(axis=tuple(axes), dtype=np.intp)


def _reduced_shape(shape: Sequence[int], axes: Sequence[int]) -> tuple[int, ...]:
    """Return ``shape`` with ``axes`` removed."""
    return tuple(length for axis, length in enumerate(shape) if axis not in set(axes))


def _reduce_coords(array: Any, axes: Sequence[int]) -> dict[str, np.ndarray]:
    """Return the coordinates that survive reducing over ``axes``.

    A coordinate is full-shape labels, so reducing an axis away only leaves it
    meaningful if it does not vary along that axis -- a per-asset sector does,
    a per-cell date does not. The ones that vary are dropped rather than being
    reduced to an arbitrary cell.
    """
    axes = set(axes)
    first = tuple(
        slice(0, 1) if axis in axes else slice(None) for axis in range(array.ndim)
    )
    kept = {}
    for name, coord in array.coords.items():
        head = coord[first]
        if arrays_equal(coord, np.broadcast_to(head, coord.shape)):
            kept[name] = head.reshape(_reduced_shape(array.shape, axes))
    return kept


def reduce(
    array: Any, op: str, dim: str | Sequence[str] | None = None, keepdims: bool = False
):
    """Reduce ``array`` over the named dimensions. See :meth:`LabelledArray.reduce`."""
    _check_op(op)
    names = list(array.dims.names) if dim is None else as_name_list(dim)
    array.dims._require(names)
    axes = tuple(array.dims.axis(name) for name in names)
    if not axes:
        return array._replace()

    backend = array.backend
    counts = _counts(array, axes)
    populated = counts > 0
    shape = _reduced_shape(array.shape, axes)

    data = {}
    for name, values in array.data.items():
        # identity() is what refuses a dtype the reduction has no meaning for,
        # so it is called whether or not there is a mask to apply it to.
        filler = identity(op, backend.dtype(values))
        if array.mask is not None:
            values = backend.where(array.mask, values, filler)
        reduced = backend.reduce(values, _BASE_OP[op], axes)
        if op == "mean":
            reduced = reduced / backend.asarray(np.maximum(counts, 1))
        if not populated.all():
            reduced = backend.where(
                populated, reduced, default_fill_value(backend.dtype(reduced))
            )
        data[name] = backend.reshape(reduced, shape)

    coords = _reduce_coords(array, axes)
    mask = None if populated.all() else populated.reshape(shape)
    dims = array.dims.drop(names)
    reduced_array = array._replace(dims=dims, data=data, coords=coords, mask=mask)
    if not keepdims:
        return reduced_array
    return _restore(array, reduced_array, names)


def _restore(array: Any, reduced: Any, names: Sequence[str]):
    """Put the reduced dimensions back, each one entry long, in their old places."""
    reduced_names = set(names)
    shape = tuple(
        1 if name in reduced_names else length
        for name, length in zip(array.dims.names, array.shape, strict=False)
    )
    labels = {
        name: array.dims[name][:1] if name in reduced_names else reduced.dims[name]
        for name in array.dims.names
    }
    dims = type(array.dims)(labels, natural=array.dims.natural)
    backend = array.backend
    return reduced._replace(
        dims=dims,
        data={
            name: backend.reshape(value, shape) for name, value in reduced.data.items()
        },
        coords={name: coord.reshape(shape) for name, coord in reduced.coords.items()},
        mask=None if reduced.mask is None else reduced.mask.reshape(shape),
    )


def _coord_grouping(array: Any, name: str) -> tuple[np.ndarray, int]:
    """Return the per-label values of coordinate ``name`` and the axis it varies along."""
    coord = array.coords[name]
    varies = [
        axis
        for axis in range(array.ndim)
        if not arrays_equal(
            coord,
            np.broadcast_to(
                coord[
                    tuple(
                        slice(0, 1) if a == axis else slice(None)
                        for a in range(array.ndim)
                    )
                ],
                coord.shape,
            ),
        )
    ]
    if len(varies) != 1:
        raise DimensionError(
            f"cannot group by coordinate {name!r}: it varies along "
            f"{[array.dims.names[axis] for axis in varies] or 'no dimension'}, and grouping "
            f"needs a coordinate that varies along exactly one"
        )
    axis = varies[0]
    picked = tuple(slice(None) if a == axis else 0 for a in range(array.ndim))
    return np.asarray(coord[picked]), axis


class GroupBy:
    """One dimension split into groups, and the reductions over them.

    Returned by :meth:`~labelled_array.LabelledArray.groupby`; not built
    directly. Reducing returns an array whose grouping dimension is labelled by
    the distinct group values, in the order :func:`labelled_array._reshape.factorize`
    puts them -- sorted, where the dtype allows it.
    """

    def __init__(self, array: Any, dim: str, values: np.ndarray) -> None:
        self._array = array
        self._dim = dim
        self._labels, self._codes = factorize(values)

    @property
    def dim(self) -> str:
        """Dimension being grouped."""
        return self._dim

    @property
    def labels(self) -> np.ndarray:
        """The distinct group values, in the order the result is labelled by."""
        return self._labels

    @property
    def groups(self) -> dict[Any, np.ndarray]:
        """Mapping of group value to the positions along the dimension it holds."""
        return {
            label: np.flatnonzero(self._codes == position)
            for position, label in enumerate(self._labels.tolist())
        }

    def __len__(self) -> int:
        """Number of groups."""
        return len(self._labels)

    def __iter__(self):
        """Iterate over ``(group value, sub-array)`` pairs."""
        for position, label in enumerate(self._labels.tolist()):
            yield (
                label,
                self._array.isel({self._dim: np.flatnonzero(self._codes == position)}),
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({type(self._array).__name__} over {self._dim!r}, "
            f"{len(self._labels)} groups)"
        )

    def reduce(self, op: str):
        """Reduce each group over the grouping dimension.

        Parameters
        ----------
        op : {'sum', 'mean', 'min', 'max'}
            Reduction to apply within each group.

        Returns:
        -------
        LabelledArray
            The array with the grouping dimension relabelled by group value.
        """
        _check_op(op)
        pieces = []
        for position, _label in enumerate(self._labels):
            positions = np.flatnonzero(self._codes == position)
            group = self._array.isel({self._dim: positions})
            reduced = reduce(group, op, [self._dim], keepdims=True)
            pieces.append(
                reduced._replace(
                    dims=reduced.dims.replace(
                        self._dim, self._labels[position : position + 1], natural=False
                    )
                )
            )
        shared = set.intersection(*(set(piece.coords) for piece in pieces))
        pieces = [
            piece._replace(
                coords={
                    name: value
                    for name, value in piece.coords.items()
                    if name in shared
                }
            )
            for piece in pieces
        ]
        return concat(pieces, self._dim, join="exact")

    def sum(self):
        """Sum each group. See :meth:`reduce`."""
        return self.reduce("sum")

    def mean(self):
        """Average each group. See :meth:`reduce`."""
        return self.reduce("mean")

    def min(self):
        """Smallest value in each group. See :meth:`reduce`."""
        return self.reduce("min")

    def max(self):
        """Largest value in each group. See :meth:`reduce`."""
        return self.reduce("max")


def groupby(array: Any, by: Any, dim: str | None = None) -> GroupBy:
    """Group one dimension. See :meth:`LabelledArray.groupby`."""
    if isinstance(by, str):
        if by in array.coords:
            values, axis = _coord_grouping(array, by)
            found = array.dims.names[axis]
            if dim is not None and dim != found:
                raise DimensionError(
                    f"coordinate {by!r} varies along {found!r}, not along {dim!r}"
                )
            return GroupBy(array, found, values)
        if by in array.dims:
            raise DimensionError(
                f"{by!r} is a dimension, and its labels are unique, so grouping by them "
                "would leave every group one entry long; pass the group of each label instead"
            )
        raise DimensionError(
            f"unknown coordinate {by!r}, known: {sorted(array.coords)}"
        )

    if dim is None:
        raise DimensionError(
            "dim must be given when grouping by an array of group values"
        )
    array.dims._require([dim])
    values = as_1d(by, dim, "group values")
    if len(values) != len(array.dims[dim]):
        raise DimensionError(
            f"got {len(values)} group values for dimension {dim!r} of length "
            f"{len(array.dims[dim])}"
        )
    return GroupBy(array, dim, values)


class Rolling:
    """Trailing windows along one dimension, and the reductions over them.

    Returned by :meth:`~labelled_array.LabelledArray.rolling`; not built
    directly. The window at each position covers that position and the
    ``window - 1`` before it, so the result keeps the shape, the labels and the
    coordinates of the array it came from.
    """

    def __init__(
        self, array: Any, dim: str, window: int, min_periods: int | None = None
    ) -> None:
        array.dims._require([dim])
        length = len(array.dims[dim])
        if window < 1 or window > length:
            raise DimensionError(
                f"window must be between 1 and the length of dimension {dim!r} "
                f"({length}), got {window}"
            )
        if min_periods is not None and not 1 <= min_periods <= window:
            raise ValueError(
                f"min_periods must be between 1 and {window}, got {min_periods}"
            )
        self._array = array
        self._dim = dim
        self._window = window
        self._min_periods = window if min_periods is None else min_periods

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({type(self._array).__name__} over {self._dim!r}, "
            f"window={self._window}, min_periods={self._min_periods})"
        )

    def _counts(self, axis: int) -> np.ndarray:
        """Return how many populated cells each window holds.

        Without a mask every cell is populated, so the count depends only on
        how far along the axis the window sits -- one vector, broadcast, rather
        than a second pass over every cell of the array.
        """
        array = self._array
        length = array.shape[axis]
        if array.mask is None:
            along = np.minimum(np.arange(1, length + 1), self._window)
            spread = [1] * array.ndim
            spread[axis] = length
            return along.reshape(spread)
        numpy_backend = _numpy_backend()
        padded = self._pad(numpy_backend, array.mask.astype(np.intp), 0, np.intp, axis)
        return numpy_backend.sliding_window(padded, axis, self._window).sum(axis=-1)

    def _pad(self, backend: Any, values: Any, fill: Any, dtype: Any, axis: int) -> Any:
        """Prepend ``window - 1`` filler entries so every position has a window."""
        if self._window == 1:
            return values
        shape = list(backend.shape(values))
        shape[axis] = self._window - 1
        return backend.concatenate(
            [backend.full(shape, fill, dtype), values], axis=axis
        )

    def reduce(self, op: str):
        """Reduce each trailing window.

        Parameters
        ----------
        op : {'sum', 'mean', 'min', 'max'}
            Reduction to apply within each window.

        Returns:
        -------
        LabelledArray
            The array's shape and labels, with each cell holding the reduction
            over its window. Cells whose window holds fewer than
            ``min_periods`` populated entries are filled and marked absent.
        """
        _check_op(op)
        array, backend = self._array, self._array.backend
        axis = array.dims.axis(self._dim)
        counts = self._counts(axis)
        enough = counts >= self._min_periods

        data = {}
        for name, values in array.data.items():
            dtype = backend.dtype(values)
            filler = identity(op, dtype)
            if array.mask is not None:
                values = backend.where(array.mask, values, filler)
            windows = backend.sliding_window(
                self._pad(backend, values, filler, dtype, axis), axis, self._window
            )
            rolled = backend.reduce(windows, _BASE_OP[op], [-1])
            if op == "mean":
                rolled = rolled / backend.asarray(np.maximum(counts, 1))
            if not enough.all():
                rolled = backend.where(
                    enough, rolled, default_fill_value(backend.dtype(rolled))
                )
            data[name] = rolled

        if enough.all():
            return array._replace(data=data, mask=None)
        return array._replace(
            data=data, mask=np.broadcast_to(enough, array.shape).copy()
        )

    def sum(self):
        """Sum each window. See :meth:`reduce`."""
        return self.reduce("sum")

    def mean(self):
        """Average each window. See :meth:`reduce`."""
        return self.reduce("mean")

    def min(self):
        """Smallest value in each window. See :meth:`reduce`."""
        return self.reduce("min")

    def max(self):
        """Largest value in each window. See :meth:`reduce`."""
        return self.reduce("max")


def _numpy_backend():
    """Return the numpy backend, which the mask and the counts always live on."""
    from labelled_array.backends import get_backend

    return get_backend("numpy")


def rolling(
    array: Any, dim: str, window: int, min_periods: int | None = None
) -> Rolling:
    """Return trailing windows along ``dim``. See :meth:`LabelledArray.rolling`."""
    return Rolling(array, dim, window, min_periods=min_periods)
