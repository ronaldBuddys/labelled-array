"""pandas DataFrame adapter."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from labelled_array.adapters.base import DataFrameAdapter

__all__ = ["PandasAdapter"]


class PandasAdapter(DataFrameAdapter):
    """Adapter for :class:`pandas.DataFrame`.

    Dimension columns may be supplied either as columns or as the frame's
    (Multi)Index, and :meth:`build_frame` puts them back into a MultiIndex
    unless ``index`` is False.
    """

    name = "pandas"
    library = "pandas"

    @classmethod
    def _instance_type(cls) -> type:
        return importlib.import_module("pandas").DataFrame

    @property
    def _pd(self):
        return importlib.import_module("pandas")

    def n_rows(self, df: Any) -> int:
        return len(df)

    def index_names(self, df: Any) -> list[str]:
        if df.index.nlevels == 1 and df.index.name is None:
            return []
        return [name for name in df.index.names if name is not None]

    def reset_index(self, df: Any) -> Any:
        names = self.index_names(df)
        return df.reset_index(level=names) if names else df

    def build_frame(
        self,
        columns: dict[str, np.ndarray],
        index_names: Sequence[str] = (),
        index: bool = True,
    ) -> Any:
        df = self._pd.DataFrame(columns, copy=False)
        if index and index_names:
            df = df.set_index(list(index_names))
        return df
