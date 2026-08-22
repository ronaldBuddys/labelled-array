"""Numpy array backend, the package default."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from labelled_array.backends.base import ArrayBackend

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
