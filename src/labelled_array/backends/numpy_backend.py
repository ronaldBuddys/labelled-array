"""Numpy array backend, the package default."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from labelled_array.backends.base import REDUCTIONS, ArrayBackend

__all__ = ["NumpyBackend"]


class NumpyBackend(ArrayBackend):
    """Backend for :class:`numpy.ndarray`, supporting every numpy dtype."""

    name = "numpy"
    library = "numpy"

    @classmethod
    def _instance_type(cls) -> type:
        return np.ndarray

    def asarray(self, obj: Any) -> np.ndarray:
        return np.asarray(obj)

    def to_numpy(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def astype(self, arr: np.ndarray, dtype: Any) -> np.ndarray:
        return arr.astype(dtype)

    def copy(self, arr: np.ndarray) -> np.ndarray:
        return arr.copy()

    def dtype(self, arr: np.ndarray) -> np.dtype:
        return arr.dtype

    def supports_dtype(self, dtype: Any) -> bool:
        return True

    def take(self, arr: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
        return np.take(arr, indices, axis=axis)

    def transpose(self, arr: np.ndarray, axes: Sequence[int]) -> np.ndarray:
        return np.transpose(arr, axes)

    def reshape(self, arr: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        return np.reshape(arr, tuple(shape))

    def broadcast_to(self, arr: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Return a read-only zero-stride view, so broadcasting costs no memory."""
        return np.broadcast_to(arr, tuple(shape))

    def take_multi(
        self, arr: np.ndarray, indexers: Mapping[int, np.ndarray]
    ) -> np.ndarray:
        """Gather every axis in one pass, so no intermediate array is built."""
        if not indexers:
            return arr
        grids = [
            indexers[axis] if axis in indexers else np.arange(length)
            for axis, length in enumerate(arr.shape)
        ]
        return arr[np.ix_(*grids)]

    def take_along_axis(
        self, arr: np.ndarray, indices: np.ndarray, axis: int
    ) -> np.ndarray:
        return np.take_along_axis(arr, indices, axis)

    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any) -> np.ndarray:
        return np.full(tuple(shape), fill_value, dtype=np.dtype(dtype))

    def where(self, condition: np.ndarray, arr: np.ndarray, other: Any) -> np.ndarray:
        return np.where(condition, arr, other)

    def concatenate(self, arrays: Sequence[np.ndarray], axis: int) -> np.ndarray:
        return np.concatenate(list(arrays), axis=axis)

    def reduce(self, arr: np.ndarray, op: str, axes: Sequence[int]) -> np.ndarray:
        if not axes:
            return arr
        return REDUCTIONS[op](arr, axis=tuple(axes))

    def sliding_window(self, arr: np.ndarray, axis: int, window: int) -> np.ndarray:
        """Return a read-only view of the windows, costing no memory at all."""
        return np.lib.stride_tricks.sliding_window_view(arr, window, axis=axis)
