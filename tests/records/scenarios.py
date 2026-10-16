"""The calls recorded: deterministic runs of each checkpoint on small seeded arrays.

Inputs are built here from ``default_rng(0)`` and explicit dates, never from files or
the lake. Every call goes through the public ``LabelledArray`` API, which reaches the
module-level implementation (``_align.reindex`` and the like) through the module
attribute, so that it hits the recorder ``checkpoints.installed`` put there. Besides a
typical case, each checkpoint gets its edge cases (a missing label, NaN, an outer
join that creates cells), and an exception is as good a record as a value.

The polars and pandas blocks are skipped with a warning when the library is missing
(for pandas, also pyarrow, which regression-records stores its frames with), so a
plain ``uv sync`` still records everything else. Record and replay in the same
environment: a store recorded with a frame library holds frames that a replay without
it cannot read.
"""

from __future__ import annotations

import importlib
import logging
import operator
from contextlib import suppress

import numpy as np

from labelled_array import LabelledArray
from labelled_array.exceptions import DimensionError, LabelNotFoundError

log = logging.getLogger(__name__)

#: Dates the time dimension is labelled with, from an explicit start (never "today").
DATES = np.arange("2024-01-02", "2024-01-08", dtype="datetime64[D]")


def two_d(seed: int = 0) -> LabelledArray:
    """Return a ``(date, sym)`` array of two float variables over datetime labels."""
    rng = np.random.default_rng(seed)
    data = {"px": rng.normal(100.0, 5.0, (6, 3)), "vol": rng.uniform(0.1, 0.4, (6, 3))}
    return LabelledArray(data, dims={"date": DATES, "sym": ["B", "A", "C"]})


def reversed_axes(seed: int = 1) -> LabelledArray:
    """Return a ``(b, a)`` array: axes in non-alphabetical order, with int labels.

    regression-records writes dicts with sorted keys, so this is the case that proves
    a recorded ``Dimensions`` comes back in axis order.
    """
    rng = np.random.default_rng(seed)
    return LabelledArray(
        {"w": rng.normal(size=(3, 2))}, dims={"b": [30, 10, 20], "a": ["y", "x"]}
    )


def three_d(seed: int = 2) -> LabelledArray:
    """Return a ``(expiry, strike, kind)`` array with string, int and float labels."""
    rng = np.random.default_rng(seed)
    dims = {"expiry": ["1M", "3M"], "strike": [90, 100, 110], "kind": ["c", "p"]}
    return LabelledArray({"iv": rng.uniform(0.1, 0.5, (2, 3, 2))}, dims=dims)


def with_gaps(seed: int = 3) -> LabelledArray:
    """Return ``two_d`` with NaN in ``px`` at three cells."""
    base = two_d(seed)
    px = base["px"].copy()
    px[1, 0] = px[2, 0] = px[3, 2] = np.nan
    return LabelledArray({"px": px, "vol": base["vol"]}, dims=base.dims)


def indexing() -> None:
    """Select by label, list and slice, a missing label; isel by position."""
    la, ba, cube = two_d(), reversed_axes(), three_d()
    la.select({"sym": "A"})
    la.select({"sym": ["C", "A"]})
    la.select({"sym": "A"}, drop=True)
    la.select({"date": DATES[1]}, drop=True)
    la.select({"date": DATES[[4, 0]], "sym": ["B"]})
    with suppress(LabelNotFoundError):  # labels, not a slice, so refused as unknown
        la.select({"date": slice(DATES[1], DATES[3])})
    with suppress(LabelNotFoundError):
        la.select({"sym": "Z"})
    la.select({"sym": ["A", "Z"]}, missing="ignore")
    ba.select({"b": [20, 30]})
    cube.select({"strike": [110, 90], "kind": "p"}, drop=True)
    la.isel({"date": -1})
    la.isel({"date": -2}, drop=True)
    la.isel({"date": slice(1, None, 2)})
    la.isel({"date": np.array([3, 0, 5]), "sym": np.array([-1, 0])})
    ba.isel({"a": [1, 0]})
    cube.isel({"strike": -1, "expiry": 0}, drop=True)
    broadcast_slicing()


def broadcast_slicing() -> None:
    """Slice a variable stored broadcast along ``strike`` (zero stride) and others."""
    rng = np.random.default_rng(4)
    dims = {"expiry": ["1M", "3M"], "strike": [90, 100, 110], "kind": ["c", "p"]}
    data = {
        "iv": rng.uniform(0.1, 0.5, (2, 3, 2)),
        "tau": rng.uniform(0.1, 1, (2, 1, 1)),
    }
    bc = LabelledArray(data, dims=dims, broadcast=True)
    bc.isel({"strike": [2, 0]})
    bc.isel({"strike": slice(0, 2), "kind": -1})
    bc.isel({"expiry": 1, "strike": 1}, drop=True)
    bc.select({"strike": [110, 90]})
    bc.select({"strike": 100, "kind": "p"}, drop=True)


def alignment() -> None:
    """Reindex with new labels, ffill across them, align and concat."""
    la, gaps, ba = two_d(), with_gaps(), reversed_axes()
    new_dates = np.array(
        ["2024-01-03", "2024-01-10", "2024-01-02"], dtype="datetime64[D]"
    )
    wider = la.reindex({"date": new_dates, "sym": ["A", "D"]})  # NaN-filled cells
    la.reindex({"date": new_dates}, fill_value=0.0)
    ba.reindex({"b": [10, 99, 30]})
    wider.ffill("date")
    wider.ffill("date", limit=1)
    la.reindex({"date": DATES[:2]}).ffill("sym")
    gaps.ffill("date")
    gaps.ffill("date", limit=1)

    left, right = la.isel({"date": slice(0, 4)}), la.isel({"date": slice(2, 6)})
    left.align(right)
    left.align(right, join="outer")
    left.align(right, join="left", fill_value=-1.0)
    left.align(right.select({"sym": ["A", "C"]}), join="outer")
    with suppress(DimensionError):  # exact join on differing labels raises
        left.align(right, join="exact")

    # Along an existing dimension, then a new one.
    tail = la.isel({"date": slice(4, None)})
    LabelledArray.concat([left, tail], "date")
    LabelledArray.concat([left, tail.select({"sym": ["A", "C"]})], "date", join="outer")
    LabelledArray.concat([la, la * 2.0], "run", labels=["base", "scaled"])
    LabelledArray.concat([ba, ba + 1.0], "step", labels=[0, 1])


def grouping_and_windows() -> None:
    """Group and roll, with sum/mean/min/max, and a NaN in the window."""
    la, gaps, cube = two_d(), with_gaps(), three_d()
    by_sym = ["x", "y", "x"]
    for op in ("sum", "mean", "min", "max"):
        la.groupby(by_sym, dim="sym").reduce(op)
    la.groupby(by_sym, dim="sym").sum()
    la.groupby(by_sym, dim="sym").mean()
    gaps.groupby(["u", "u", "v", "v", "u", "v"], dim="date").mean()
    cube.groupby(["lo", "hi", "hi"], dim="strike").sum()
    with suppress(DimensionError):  # wrong group length
        la.groupby(["x", "y"], dim="sym")

    la.rolling(dim="date", window=3).mean()
    la.rolling(dim="date", window=3).sum()
    gaps.rolling(dim="date", window=3).mean()
    gaps.rolling(dim="date", window=3, min_periods=1).mean()
    gaps.rolling(dim="date", window=2).max()
    la.rolling(dim="date", window=3).reduce("min")


def arithmetic() -> None:
    """Add, subtract, multiply and divide by a scalar, a broadcast and an aligned array."""
    la, ba = two_d(), reversed_axes()
    weights = np.array([1.0, 2.0, 4.0])  # numpy broadcasts it along the last axis
    shifted = la.isel({"date": slice(2, 6)}).select({"sym": ["C", "B", "A"]})
    zeros = np.zeros(la.shape)
    zero = LabelledArray({"px": zeros, "vol": zeros}, dims=la.dims)
    for op in (operator.add, operator.sub, operator.mul, operator.truediv):
        op(la, 2.0)
        op(la, weights)  # an ndarray broadcast over date
        op(la, shifted)  # aligned: inner join on date, sym reordered
        op(la, shifted.transpose(["sym", "date"]))  # axes swapped, then aligned
        op(ba, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        la / zero  # division by zero: inf and NaN, not an error
    with suppress(ValueError):  # a variable the other side lacks
        la + LabelledArray({"other": np.ones(3)}, dims={"sym": ["B", "A", "C"]})
    with suppress(ValueError):  # a dimension the other side lacks
        la + LabelledArray(
            {"px": np.ones(3), "vol": np.ones(3)}, dims={"sym": list("BAC")}
        )


def _available(*modules: str) -> bool:
    """Return whether every module imports, warning about the first that does not."""
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            log.warning("%s is not installed; skipping the DataFrame scenarios", module)
            return False
    return True


def frames(engine: str) -> None:
    """Convert to and from a DataFrame with ``engine`` ('polars' or 'pandas')."""
    la, gaps = two_d(), with_gaps()
    for array in (la, gaps, three_d(), reversed_axes()):
        frame = array.to_dataframe(engine=engine)
        LabelledArray.from_dataframe(frame, dims=list(array.dims.names), engine=engine)
    gaps.to_dataframe(engine=engine, dropna=True)
    la.to_dataframe(engine=engine, index=False)
    long = la.to_dataframe(engine=engine, dropna=True)
    LabelledArray.from_dataframe(long, cols="px", dims=["date", "sym"], engine=engine)
    LabelledArray.from_dataframe(
        long, dims=["sym", "date"], fill_value=0.0, engine=engine
    )


def run_all() -> None:
    """Make every recorded call."""
    indexing()
    alignment()
    grouping_and_windows()
    arithmetic()
    if _available("polars"):
        frames("polars")
    # regression-records writes a pandas frame as parquet, which needs pyarrow (its
    # `pandas` extra); without it the pandas shim cannot be recorded.
    if _available("pandas", "pyarrow"):
        frames("pandas")
