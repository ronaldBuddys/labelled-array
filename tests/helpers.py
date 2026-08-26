"""Engine-independent helpers for the DataFrame tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def build_frame(engine: str, data: Mapping[str, Any]) -> Any:
    """Build a DataFrame of the given engine from a mapping of columns."""
    if engine == "pandas":
        import pandas as pd

        return pd.DataFrame(dict(data))
    if engine == "polars":
        import polars as pl

        return pl.DataFrame(
            {name: pl.Series(name, values) for name, values in data.items()}
        )
    raise ValueError(f"unknown engine {engine!r}")


def assert_frames_equal(engine: str, left: Any, right: Any) -> None:
    """Assert two frames of the given engine are equal, including dtypes."""
    if engine == "pandas":
        import pandas as pd

        pd.testing.assert_frame_equal(left, right)
    else:
        from polars.testing import assert_frame_equal

        assert_frame_equal(left, right)


def sort_frame(engine: str, df: Any, by: Sequence[str]) -> Any:
    """Return ``df`` sorted by ``by``, with a fresh row order."""
    if engine == "pandas":
        return df.sort_values(list(by)).reset_index(drop=True)
    return df.sort(list(by))


def column(df: Any, name: str) -> np.ndarray:
    """Return one column of ``df`` as a numpy array."""
    return np.asarray(df[name].to_numpy())


def columns(df: Any) -> list[str]:
    """Return the column names of ``df``."""
    return list(df.columns)


def build_both(data: Mapping[str, Any], dims: Sequence[str], **kwargs: Any) -> list:
    """Return the labelled array built from each engine's frame of ``data``."""
    from labelled_array import LabelledArray

    return [
        LabelledArray.from_dataframe(build_frame(engine, data), dims=dims, **kwargs)
        for engine in ("pandas", "polars")
    ]


def n_rows(engine: str, df: Any) -> int:
    """Return the number of rows in ``df``."""
    return len(df) if engine == "pandas" else df.height
