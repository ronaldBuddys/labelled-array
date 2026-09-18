"""The DataFrame operations :mod:`labelled_array` requires of a DataFrame library."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from labelled_array import _reshape
from labelled_array._plugin import Plugin

__all__ = ["DataFrameAdapter"]


class DataFrameAdapter(Plugin):
    """Adapter for one DataFrame library.

    Adapters read columns as numpy arrays and assemble numpy arrays back into a
    frame; the reshaping between flat rows and a dense array is shared, so
    every library goes through the same code and produces the same result.

    The one step a library may do itself is :meth:`factorize`, where its own
    string handling beats a Python loop over an object array -- as polars' does,
    several times over. A native path must reproduce the numpy reference
    exactly, and the test suite holds it to that. pandas has none: its
    factorize is barely faster than the reference, and merges strings that
    differ only after a NUL character.

    The index-related methods default to "this library has no index", which is
    true of every DataFrame library except pandas.
    """

    @abstractmethod
    def n_rows(self, df: Any) -> int:
        """Return the number of rows in ``df``."""

    @abstractmethod
    def build_frame(
        self,
        columns: dict[str, np.ndarray],
        index_names: Sequence[str] = (),
        index: bool = True,
    ) -> Any:
        """Assemble numpy columns into a DataFrame.

        Parameters
        ----------
        columns : dict of str to numpy.ndarray
            Column name to values, in the order they should appear.
        index_names : sequence of str, optional
            Columns that identify a row.
        index : bool, default True
            Whether ``index_names`` should become the frame's index. Libraries
            with no index concept ignore this.
        """

    def column_names(self, df: Any) -> list[str]:
        """Return the column names of ``df``."""
        return list(df.columns)

    def column_values(self, df: Any, column: str) -> np.ndarray:
        """Return column ``column`` of ``df`` as a numpy array."""
        return np.asarray(df[column].to_numpy())

    def factorize(self, df: Any, column: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the unique labels of ``column`` and each row's position in them.

        Parameters
        ----------
        df : DataFrame
            Frame of this adapter's library.
        column : str
            Dimension column to factorize.

        Returns
        -------
        labels : numpy.ndarray
            The unique values, exactly as :func:`labelled_array._reshape.factorize`
            returns them: same dtype, same order.
        codes : numpy.ndarray
            Position of each row within ``labels``, as ``numpy.intp``.
        """
        native = self._native_factorize(df, column)
        if native is not None:
            return native
        return _reshape.factorize(self.column_values(df, column))

    def _native_factorize(
        self, df: Any, column: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the library's own factorization, or None to use the numpy one.

        An override must return None whenever it cannot guarantee the result is
        identical to the numpy reference -- labels, their order, their dtype and
        the codes -- so that the engine a frame came from never shows in the
        array built from it.
        """
        return None

    def index_names(self, df: Any) -> list[str]:
        """Return the names of ``df``'s index levels, empty when it has none.

        Unnamed levels are excluded, since a dimension needs a name.
        """
        return []

    def reset_index(self, df: Any) -> Any:
        """Return ``df`` with its named index levels moved into columns."""
        return df
