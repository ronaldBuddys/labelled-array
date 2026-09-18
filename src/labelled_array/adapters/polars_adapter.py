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
    def _native_factorize(
        self, df: Any, column: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pl = self._pl
        series = df[column]
        # Only strings pay: np.unique is already fast for every other dtype, and
        # a null has to reach the numpy path to be refused with the usual error.
        if series.dtype != pl.String or series.null_count():
            return None
        # polars sorts strings by their UTF-8 bytes, which is the code point
        # order Python -- and so the numpy reference -- sorts them by.
        labels = series.unique().sort()
        # An Enum's physical value is the index into its categories. Categorical
        # would not do: its physical codes come from a global string cache.
        codes = series.cast(pl.Enum(labels)).to_physical().to_numpy()
        return (
            np.asarray(labels.to_numpy(), dtype=object),
            codes.astype(np.intp, copy=False),
        )

