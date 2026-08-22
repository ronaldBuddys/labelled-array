"""PyTorch tensor backend."""

from __future__ import annotations

import importlib
from typing import Any, Sequence

import numpy as np

from labelled_array.backends.base import ArrayBackend

__all__ = ["TorchBackend"]


class TorchBackend(ArrayBackend):
    """Backend for :class:`torch.Tensor`.

    Notes
    -----
    Torch has no string, object or datetime dtypes; :meth:`supports_dtype`
    reports what the installed torch build actually accepts, so conversion of an
    unsupported variable fails loudly rather than silently coercing.
    """

    name = "torch"
    library = "torch"

    @classmethod
    def _instance_type(cls) -> type:
        return importlib.import_module("torch").Tensor

    @property
    def _torch(self):
        return importlib.import_module("torch")

    def asarray(self, obj: Any) -> Any:
        if self.handles(obj):
            return obj
        arr = np.ascontiguousarray(obj)
        self.check_dtype(arr.dtype)
        if not arr.flags.writeable:
            # from_numpy would share memory with a read-only buffer, which torch
            # warns about and cannot safely write to.
            arr = arr.copy()
        return self._torch.from_numpy(arr)

    def to_numpy(self, arr: Any) -> np.ndarray:
        return arr.detach().cpu().numpy()

    def astype(self, arr: Any, dtype: Any) -> Any:
        return arr.to(self._as_torch_dtype(dtype))

    def copy(self, arr: Any) -> Any:
        return arr.clone()

    def dtype(self, arr: Any) -> np.dtype:
        return np.dtype(str(arr.dtype).removeprefix("torch."))

    def supports_dtype(self, dtype: Any) -> bool:
        try:
            self._as_torch_dtype(dtype)
        except (TypeError, AttributeError):
            return False
        return True

    def _as_torch_dtype(self, dtype: Any):
        """Map a numpy dtype onto the equivalent torch dtype."""
        torch = self._torch
        numpy_dtype = np.dtype(dtype)
        torch_dtype = getattr(torch, numpy_dtype.name, None)
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"torch has no dtype equivalent to {numpy_dtype}")
        return torch_dtype

    def take(self, arr: Any, indices: np.ndarray, axis: int) -> Any:
        torch = self._torch
        return torch.index_select(arr, axis, torch.as_tensor(np.asarray(indices, np.int64)))

    def transpose(self, arr: Any, axes: Sequence[int]) -> Any:
        return arr.permute(tuple(axes)).contiguous()

    def reshape(self, arr: Any, shape: Sequence[int]) -> Any:
        return arr.reshape(tuple(shape))
