"""polars DataFrame adapter."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from labelled_array.adapters.base import DataFrameAdapter

__all__ = ["PolarsAdapter"]


class PolarsAdapter(DataFrameAdapter):
    """Adapter for :class:`polars.DataFrame`.

    polars has no index, so dimension columns are always ordinary columns and
    the ``index`` argument of :meth:`build_frame` is ignored.
    """

    name = "polars"
    library = "polars"

    @classmethod
    def _instance_type(cls) -> type:
        return importlib.import_module("polars").DataFrame

    @property
    def _pl(self):
        return importlib.import_module("polars")

    def n_rows(self, df: Any) -> int:
        return df.height

    def build_frame(
        self,
        columns: dict[str, np.ndarray],
        index_names: Sequence[str] = (),
        index: bool = True,
    ) -> Any:
        pl = self._pl
        return pl.DataFrame(
            {name: pl.Series(name, values) for name, values in columns.items()}
        )
