"""
Combining and reducing
======================

Arrays that carry their labels can be put side by side even when their labels
differ, and reduced along the dimensions they name. Everything in this example
is mask-aware in the same way: a cell with no source row does not contribute,
and a result with nothing behind it is marked absent rather than being invented.
"""

# %%
# The data
# --------
# A small panel of trades: three assets over four dates, with one asset that
# started late, so the array is sparse from the moment it is built.

import numpy as np
import polars as pl

from labelled_array import LabelledArray

trades = pl.DataFrame(
    {
        "asset": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB", "CCC", "CCC"],
        "date": np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"] * 2
            + ["2024-01-03", "2024-01-04"],
            dtype="datetime64[D]",
        ),
        "pnl": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0],
        "lots": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    }
)
book = LabelledArray.from_dataframe(trades, dims=["asset", "date"])

print(book)
assert book.shape == (3, 4)
assert book.density == 10 / 12, "CCC has no rows before the third"

# %%
# Reducing a dimension away
# -------------------------
# ``sum``, ``mean``, ``min`` and ``max`` take the dimensions to reduce over and
# drop them. Only populated cells count, so the mean of CCC is the mean of the
# two days it traded and not of four.

print(book.sum("date"))
print("total pnl :", book.sum("date")["pnl"].tolist())
print("mean pnl  :", book.mean("date")["pnl"].tolist())

assert book.sum("date").dims.names == ("asset",)
assert book.mean("date")["pnl"].tolist() == [2.5, 25.0, 150.0]
assert book.sum("date").dtypes["lots"] == np.dtype("int64"), (
    "an integer sum stays integer"
)

# %%
# Naming nothing reduces everything
# ---------------------------------
# The result is a zero-dimensional array -- one cell, still labelled by its
# variables.

everything = book.sum()
print(everything)
assert everything.shape == ()
assert everything["pnl"] == trades["pnl"].sum()

# %%
# A result with nothing behind it
# -------------------------------
# Reducing over a dimension where every cell is absent gives no number to
# report. Rather than a zero that cannot be told from a real one, the cell is
# filled and the mask records that it is empty.

early = book.select({"date": book.dims["date"][:2]})
totals = early.sum("date")

print(totals["pnl"], totals.mask)
assert np.isnan(totals["pnl"][2]), "CCC traded on neither of those days"
assert totals.mask.tolist() == [True, True, False]
assert len(totals.to_dataframe()) == 2, "and it is not a row on the way out"

# %%
# Grouping one dimension
# ----------------------
# ``groupby`` takes the group of each label -- as an array, or as the name of a
# coordinate holding one value per label -- and reduces within each group. The
# dimension keeps its name and takes the group values as its labels.

sectors = ["tech", "tech", "banks"]
by_sector = book.groupby(sectors, dim="asset").sum()

print(by_sector.to_dataframe())
assert by_sector.dims["asset"].tolist() == ["banks", "tech"]
assert by_sector["pnl"][1].tolist() == [11.0, 22.0, 33.0, 44.0], "AAA and BBB together"

# %%
# ...or by a coordinate
# ---------------------
# A coordinate that varies along exactly one dimension is a per-label value,
# which is what a grouping needs. Naming it is the same as passing the array.

with_sector = book._replace(
    coords={"sector": np.array([[s] * 4 for s in sectors], dtype=object)}
)
# NaN marks the cells with nothing behind them, so compare the way numpy's
# testing helpers do, with nulls matching nulls.
np.testing.assert_array_equal(
    with_sector.groupby("sector").sum()["pnl"], by_sector["pnl"]
)

# %%
# What the groups are
# -------------------
# The object itself is inspectable, and iterating gives the sub-arrays.

grouped = book.groupby(sectors, dim="asset")
print(grouped, grouped.groups)
for name, part in grouped:
    print(f"{name:6} {part.dims['asset'].tolist()}")

assert len(grouped) == 2
assert grouped.groups["tech"].tolist() == [0, 1]

# %%
# Rolling windows
# ---------------
# A window covers a position and the ones before it, so the result keeps the
# array's shape and labels. ``min_periods`` says how much data a window needs
# before it reports anything; by default it needs to be full.

rolled = book.rolling("date", 2).sum()
print(rolled["pnl"])
assert np.isnan(rolled["pnl"][0, 0]), "the first position has no window behind it"
assert rolled["pnl"][0, 1] == 1.0 + 2.0

partial = book.rolling("date", 2, min_periods=1).mean()
assert partial["pnl"][0, 0] == 1.0, "a window of one, since one is enough"
assert partial.mask[2, :2].tolist() == [False, False], "CCC still has nothing there"

# %%
# Rolling along a natural index
# -----------------------------
# The dimension this is really for is a natural index, where position means
# "the nth observation of this group" and the dates themselves are a
# coordinate. Neighbouring positions are then neighbouring observations, which
# is what a window wants, whether or not the groups share dates.

panel = LabelledArray.from_dataframe(
    trades, cols=["pnl"], dims=["asset", "date"], natural=["date"]
)
averaged = panel.rolling("date_idx", 2, min_periods=1).mean()
print(averaged["pnl"])
print(averaged.coords["date"][2], "-- the dates CCC's positions stand for")

assert averaged.dims.names == ("asset", "date_idx")
assert averaged["pnl"][2, :2].tolist() == [100.0, 150.0], "CCC's own first two trades"
assert averaged.coords["date"].shape == averaged.shape, "the dates are still there"

# Labelled by date instead, CCC's first window would have been two absent cells
# rather than its first two trades -- the same rows, a different question.
assert np.isnan(book[["pnl"]].rolling("date", 2, min_periods=1).mean()["pnl"][2, 0])

# %%
# Moving onto a different label set
# ---------------------------------
# ``reindex`` gathers the labels an array has and fills the ones it does not,
# marking them absent. It is how an array is put onto a layout decided
# somewhere else.

week = np.datetime64("2024-01-01") + np.arange(5)
padded = book.reindex(date=week)

print(padded.mask)
assert padded.shape == (3, 5)
assert np.isnan(padded["pnl"][:, 4]).all()
assert not padded.mask[:, 4].any()
assert padded.dtypes["lots"] == np.dtype("int64"), "the fill does not widen a dtype"

# %%
# ...and filling what is missing
# ------------------------------
# ``fillna`` puts a value in every absent or null cell and drops the mask, so
# the array is dense and every cell becomes a row on the way out.

filled = padded.fillna({"pnl": 0.0, "lots": 0})
assert filled.mask is None
assert filled["pnl"][2, 0] == 0.0
assert len(filled.to_dataframe()) == filled.size

# %%
# Aligning two arrays
# -------------------
# Two arrays whose labels differ are put onto one label set first. An inner
# join keeps what both have; an outer join keeps everything and marks what
# neither side filled.

benchmark = LabelledArray.from_dataframe(
    pl.DataFrame(
        {
            "asset": ["BBB", "CCC", "DDD"],
            "date": np.array(["2024-01-03"] * 3, dtype="datetime64[D]"),
            "pnl": [1.0, 2.0, 3.0],
        }
    ),
    dims=["asset", "date"],
)

inner, _ = book[["pnl"]].align(benchmark)
outer, _ = book[["pnl"]].align(benchmark, join="outer")
print("inner:", inner.dims["asset"].tolist(), inner.dims["date"].shape)
print("outer:", outer.dims["asset"].tolist(), outer.dims["date"].shape)

assert inner.dims["asset"].tolist() == ["BBB", "CCC"]
assert outer.dims["asset"].tolist() == ["AAA", "BBB", "CCC", "DDD"]
assert not outer.mask[3].any(), "DDD is a row of cells nothing filled"

# %%
# Arithmetic aligns on its own
# ----------------------------
# Operators between two arrays align first, so cells are combined by label
# rather than by position. The default is an inner join: the result covers what
# both operands actually hold.

excess = book[["pnl"]] - benchmark
print(excess.to_dataframe())
assert excess.dims["asset"].tolist() == ["BBB", "CCC"]
assert excess["pnl"].tolist() == [[30.0 - 1.0], [100.0 - 2.0]]

# %%
# Joining arrays end to end
# -------------------------
# ``concat`` extends a dimension every array has, or stacks along a new one.
# The other dimensions are aligned as part of it, so arrays that cover
# different assets can still be joined along a date.

january = book.select({"date": book.dims["date"][:2]})
later = book.select({"date": book.dims["date"][2:]})
rejoined = LabelledArray.concat([january, later], dim="date")

assert rejoined.equals(book), "the split and the join undo each other"

stacked = LabelledArray.concat([january, january], dim="run", labels=["a", "b"])
print(stacked)
assert stacked.dims.names == ("run", "asset", "date")
assert stacked.shape == (2, 3, 2)

# %%
# Where it stops
# --------------
# The operations are deliberately narrow, and say so rather than guessing:
#
# * Arithmetic aligns labels, but the dimension *names* must match -- nothing
#   here broadcasts one dimension onto another.
# * ``groupby`` groups one dimension by one value per label. A coordinate that
#   varies along two dimensions is refused rather than flattened.
# * Windows are trailing, never centred.
# * There is no ``skipna``: absence is what the mask records, and a NaN that is
#   masked in is data.

only_pnl = book[["pnl"]]
try:
    only_pnl + only_pnl.rename(asset="ticker")
except ValueError as error:
    print(error)

try:
    book.groupby("nothing")
except Exception as error:
    print(error)
