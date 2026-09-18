"""polars DataFrame adapter."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from labelled_array.adapters.base import DataFrameAdapter

__all__ = ["PolarsAdapter"]

#: numpy time units polars has no dtype for, and the nearest finer one it has.
#: Going finer is exact; calendar units land on Date rather than a timestamp.
_DATETIME_UNITS = {"Y": "D", "M": "D", "W": "D", "h": "ms", "m": "ms", "s": "ms"}
_TIMEDELTA_UNITS = {"W": "ms", "D": "ms", "h": "ms", "m": "ms", "s": "ms"}


def _polars_compatible(values: np.ndarray) -> np.ndarray:
    """Return ``values`` in a time unit polars can hold, unchanged otherwise.

    polars refuses a ``timedelta64[D]`` -- what subtracting two dates gives --
    rather than widening it as pandas does, so the widening is done here.
    """
    if values.dtype.kind not in "Mm":
        return values
    units = _DATETIME_UNITS if values.dtype.kind == "M" else _TIMEDELTA_UNITS
    unit = units.get(np.datetime_data(values.dtype)[0])
    return values if unit is None else values.astype(f"{values.dtype.kind}8[{unit}]")


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

    def build_frame(
        self,
        columns: dict[str, np.ndarray],
        index_names: Sequence[str] = (),
        index: bool = True,
    ) -> Any:
        pl = self._pl
        return pl.DataFrame(
            {
                name: pl.Series(name, _polars_compatible(values))
                for name, values in columns.items()
            }
        )
