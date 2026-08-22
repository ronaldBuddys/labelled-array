"""The array operations :mod:`labelled_array` requires of an array library."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

import numpy as np

from labelled_array._plugin import Plugin
from labelled_array.exceptions import BackendDtypeError

__all__ = ["ArrayBackend"]


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
