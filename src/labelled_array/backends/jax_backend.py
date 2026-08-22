"""JAX array backend."""

from __future__ import annotations

import importlib
from typing import Any, Sequence

import numpy as np

from labelled_array.backends.base import ArrayBackend

__all__ = ["JaxBackend"]

#: Dtype kinds JAX cannot represent (strings, bytes, objects, void, dates).
_UNSUPPORTED_KINDS = frozenset("USOVMm")


class JaxBackend(ArrayBackend):
    """Backend for :class:`jax.Array`.

    Notes
    -----
    JAX arrays are immutable and, unless ``jax_enable_x64`` is set, silently
    narrow 64-bit dtypes to 32-bit; :meth:`dtype` therefore reports the dtype
    JAX actually allocated, not the one requested.
    """

    name = "jax"
    library = "jax"

    @classmethod
    def _instance_type(cls) -> type:
        return importlib.import_module("jax").Array

    @property
    def _jnp(self):
        return importlib.import_module("jax.numpy")

    def asarray(self, obj: Any) -> Any:
        if not self.handles(obj):
            # Validate the source dtype, but let JAX pick the allocated one so
            # that its 64-bit narrowing stays silent unless explicitly asked for.
            self.check_dtype(np.asarray(obj).dtype)
        return self._jnp.asarray(obj)

    def to_numpy(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def astype(self, arr: Any, dtype: Any) -> Any:
        self.check_dtype(dtype)
        return arr.astype(dtype)

    def copy(self, arr: Any) -> Any:
        return self._jnp.array(arr, copy=True)

    def dtype(self, arr: Any) -> np.dtype:
        return np.dtype(arr.dtype)

    def supports_dtype(self, dtype: Any) -> bool:
        return np.dtype(dtype).kind not in _UNSUPPORTED_KINDS

    def take(self, arr: Any, indices: np.ndarray, axis: int) -> Any:
        return self._jnp.take(arr, self._jnp.asarray(indices), axis=axis)

    def transpose(self, arr: Any, axes: Sequence[int]) -> Any:
        return self._jnp.transpose(arr, tuple(axes))

    def reshape(self, arr: Any, shape: Sequence[int]) -> Any:
        return self._jnp.reshape(arr, tuple(shape))
