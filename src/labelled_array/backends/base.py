"""The array operations :mod:`labelled_array` requires of an array library."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from labelled_array._plugin import Plugin
from labelled_array.exceptions import BackendDtypeError

__all__ = ["ArrayBackend", "REDUCTIONS"]

#: The reductions a backend must provide, as their numpy equivalents.
REDUCTIONS = {"sum": np.sum, "min": np.min, "max": np.max}


class ArrayBackend(Plugin):
    """Adapter for one array library.

    Subclasses are the only place a concrete array library is touched, so a new
    library is supported by adding a subclass and registering it, without
    changing anything else in the package. The methods below are the complete
    set of array operations the package needs.
    """

    # -- conversion --------------------------------------------------------------

    @abstractmethod
    def asarray(self, obj: Any) -> Any:
        """Return ``obj`` as an array of this backend, converting if needed."""

    @abstractmethod
    def to_numpy(self, arr: Any) -> np.ndarray:
        """Return ``arr`` as a numpy array."""

    @abstractmethod
    def astype(self, arr: Any, dtype: Any) -> Any:
        """Return ``arr`` cast to ``dtype`` (given as a numpy dtype)."""

    @abstractmethod
    def copy(self, arr: Any) -> Any:
        """Return an independent copy of ``arr``."""

    # -- inspection --------------------------------------------------------------

    @abstractmethod
    def dtype(self, arr: Any) -> np.dtype:
        """Return the dtype of ``arr`` as a numpy dtype."""

    @abstractmethod
    def supports_dtype(self, dtype: Any) -> bool:
        """Whether this backend can hold data of the numpy dtype ``dtype``."""

    def shape(self, arr: Any) -> tuple[int, ...]:
        """Return the shape of ``arr``."""
        return tuple(arr.shape)

    # -- shape manipulation ------------------------------------------------------

    @abstractmethod
    def take(self, arr: Any, indices: np.ndarray, axis: int) -> Any:
        """Return the entries of ``arr`` at integer ``indices`` along ``axis``."""

    @abstractmethod
    def transpose(self, arr: Any, axes: Sequence[int]) -> Any:
        """Return ``arr`` with its axes permuted to ``axes``."""

    @abstractmethod
    def reshape(self, arr: Any, shape: Sequence[int]) -> Any:
        """Return ``arr`` reshaped to ``shape``."""

    # -- provided implementations ------------------------------------------------

    # Everything below has a working implementation in terms of the methods
    # above, or by way of numpy, so an existing backend keeps working when the
    # package learns a new operation. Overriding one is a performance choice.

    def take_multi(self, arr: Any, indexers: Mapping[int, np.ndarray]) -> Any:
        """Return the entries at integer indices along several axes at once.

        Parameters
        ----------
        arr : array
            Array to gather from.
        indexers : mapping of int to ndarray of int
            Indices to keep, per axis. Axes not named are kept whole.

        Notes
        -----
        The default applies one axis at a time, materialising an intermediate
        array per axis. A backend able to gather several axes in one pass --
        numpy's ``np.ix_`` -- should override this.
        """
        for axis, indices in indexers.items():
            arr = self.take(arr, indices, axis)
        return arr

    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any) -> Any:
        """Return a new array of ``shape`` filled with ``fill_value``."""
        return self.asarray(np.full(tuple(shape), fill_value, dtype=np.dtype(dtype)))

    def where(self, condition: np.ndarray, arr: Any, other: Any) -> Any:
        """Return ``arr`` where ``condition`` holds and ``other`` elsewhere.

        ``condition`` is a numpy boolean array broadcastable against ``arr``,
        because the presence mask is always numpy; ``other`` is a scalar of a
        dtype ``arr`` can hold.
        """
        return self.asarray(np.where(condition, self.to_numpy(arr), other))

    def concatenate(self, arrays: Sequence[Any], axis: int) -> Any:
        """Return ``arrays`` joined along an existing ``axis``."""
        return self.asarray(
            np.concatenate([self.to_numpy(arr) for arr in arrays], axis=axis)
        )

    def reduce(self, arr: Any, op: str, axes: Sequence[int]) -> Any:
        """Reduce ``arr`` over ``axes``.

        Parameters
        ----------
        arr : array
            Array to reduce.
        op : {'sum', 'min', 'max'}
            Reduction to apply. Missing cells are substituted with the
            operation's identity before this is called, so the reduction
            itself has nothing to skip.
        axes : sequence of int
            Axes to reduce over; empty reduces nothing.
        """
        if not axes:
            return arr
        return self.asarray(REDUCTIONS[op](self.to_numpy(arr), axis=tuple(axes)))

    def sliding_window(self, arr: Any, axis: int, window: int) -> Any:
        """Return the windows of length ``window`` along ``axis``, as a new last axis.

        The result has ``shape[axis]`` replaced by ``shape[axis] - window + 1``
        and one axis of length ``window`` appended, so reducing over ``-1``
        gives one value per complete window.

        Notes
        -----
        The default builds the windows by stacking shifted copies, which costs
        ``window`` times the memory of the input. A backend that can express
        them as a view -- numpy's ``sliding_window_view``, torch's ``unfold`` --
        should override this.
        """
        length = self.shape(arr)[axis]
        if window > length:
            raise ValueError(f"window {window} is longer than axis {axis} of length {length}")
        kept = length - window + 1
        parts = []
        for offset in range(window):
            shifted = self.take(arr, np.arange(offset, offset + kept), axis)
            parts.append(self.reshape(shifted, (*self.shape(shifted), 1)))
        return self.concatenate(parts, axis=-1)

    def check_dtype(self, dtype: Any, context: str = "") -> None:
        """Raise :class:`BackendDtypeError` if this backend cannot hold ``dtype``.

        Parameters
        ----------
        dtype : numpy dtype
            Dtype to check.
        context : str, optional
            Name of the variable being converted, used in the error message.
        """
        if not self.supports_dtype(dtype):
            where = f" for {context!r}" if context else ""
            raise BackendDtypeError(
                f"backend {self.name!r} cannot hold dtype {np.dtype(dtype)}{where}"
            )
