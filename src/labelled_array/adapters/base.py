"""The DataFrame operations :mod:`labelled_array` requires of a DataFrame library."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

import numpy as np

from labelled_array._plugin import Plugin

__all__ = ["DataFrameAdapter"]


class DataFrameAdapter(Plugin):
    """Adapter for one DataFrame library.

    Adapters only read columns as numpy arrays and assemble numpy arrays back
    into a frame; the reshaping between flat rows and a dense array is shared,
    so every library goes through the same code and produces the same result.

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

    def index_names(self, df: Any) -> list[str]:
        """Return the names of ``df``'s index levels, empty when it has none.

        Unnamed levels are excluded, since a dimension needs a name.
        """
        return []

    def reset_index(self, df: Any) -> Any:
        """Return ``df`` with its named index levels moved into columns."""
        return df
