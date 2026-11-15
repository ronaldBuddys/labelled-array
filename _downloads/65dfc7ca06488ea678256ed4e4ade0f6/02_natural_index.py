"""
.. _natural_index_example:

``from_dataframe(natural=...)`` : natural indices and sparsity
==============================================================

A dense array spans the product of its dimension labels, which is wasteful when
one dimension's labels are nearly unique per group -- dates, most of the time.
Replacing that dimension with a natural index collapses the array while keeping
the original values, so the round trip is still exact.
"""

# %%
# The problem: a date per row
# ---------------------------
# Three groups of four rows, each with dates that appear in no other group.
# A ``datetime64[D]`` array becomes a polars ``Date`` column.

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from labelled_array import DuplicateIndexError, LabelledArray

panel = pl.DataFrame(
    {
        "group": np.repeat(["A", "B", "C"], 4),
        "item": list("wxyz") * 3,
        "date": np.arange(12).astype("datetime64[D]"),
        "value": np.arange(12, dtype=float),
    }
)
assert panel.schema["date"] == pl.Date
print(panel.head())

# %%
# Labelling the dates directly
# ----------------------------
# Every distinct date becomes a label of its own, so the array spans
# 3 x 4 x 12 cells to hold twelve numbers.

labelled = LabelledArray.from_dataframe(
    panel, cols=["value"], dims=["group", "item", "date"]
)

print("shape  :", labelled.shape)
print("cells  :", labelled.size)
print("density:", f"{labelled.density:.1%}")

assert labelled.shape == (3, 4, 12)
assert labelled.density < 0.1

# %%
# Indexing the dates by position
# ------------------------------
# ``natural=["date"]`` numbers the dates within each combination of the other
# dimensions instead of labelling them. The dimension becomes ``date_idx``,
# labelled ``0 .. n-1``, and the array is dense.

natural = LabelledArray.from_dataframe(
    panel, cols=["value"], dims=["group", "item", "date"], natural=["date"]
)

print("shape  :", natural.shape)
print("cells  :", natural.size)
print("density:", f"{natural.density:.1%}")

assert natural.shape == (3, 4, 1)
assert natural.size < labelled.size
assert natural.dims.names == ("group", "item", "date_idx")
assert natural.dims.is_natural("date_idx")

# %%
# The dates are kept as a coordinate
# ----------------------------------
# The values the index stands in for are stored as an array of the same shape,
# so nothing is thrown away -- which is what makes the round trip work.

print(natural.coords["date"].ravel()[:4])
assert natural.coords["date"].shape == natural.shape
assert natural.coords["date"].dtype.kind == "M"

# %%
# The round trip is still exact
# -----------------------------
# ``to_dataframe`` writes the coordinate back out under its original name, and
# the dates come back as a polars ``Date`` column.

out = natural.to_dataframe()
assert_frame_equal(panel.sort(["group", "item", "date"]), out)
print(out.head())

# %%
# ...or the index itself, if that is what you want
# ------------------------------------------------
# ``expand_natural=False`` keeps the position as a column beside the dates.

positions = natural.to_dataframe(expand_natural=False)
assert positions.columns == ["group", "item", "date_idx", "date", "value"]
print(positions.head(3))

# %%
# The dimension order does not matter
# -----------------------------------
# Grouping ignores where the natural dimension sits, so the collapsed size is
# the same either way.

other_order = LabelledArray.from_dataframe(
    panel, cols=["value"], dims=["date", "group", "item"], natural=["date"]
)
assert other_order.size == natural.size

# %%
# Ragged groups
# -------------
# When groups have different numbers of dates the array takes the largest, and
# the mask marks the short ones -- still exact on the way back.

ragged = pl.DataFrame(
    {
        "group": ["A", "A", "A", "B"],
        "date": np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"],
            dtype="datetime64[D]",
        ),
        "value": [1.0, 2.0, 3.0, 4.0],
    }
)
uneven = LabelledArray.from_dataframe(
    ragged, cols=["value"], dims=["group", "date"], natural=["date"]
)
print(uneven.mask)
assert uneven.shape == (2, 3)
assert uneven.density == 4 / 6
assert_frame_equal(ragged.sort(["group", "date"]), uneven.to_dataframe())

# %%
# A ladder shared by the rows beneath it
# --------------------------------------
# Numbering rows needs labels that are unique within the group. A long option
# chain breaks that on purpose: there is one row per strike, so an expiry
# repeats within its trade date, and numbering the rows would count every
# strike as an expiry of its own. ``from_dataframe`` refuses rather than build
# that ladder.

chain = pl.DataFrame(
    {
        "trade_date": ["2024-06-03"] * 5 + ["2024-06-04"] * 2,
        "expiry": [0.1, 0.1, 0.1, 0.5, 0.5, 0.4, 0.4],
        "strike": [90.0, 100.0, 110.0, 95.0, 105.0, 80.0, 85.0],
        "vol": [0.22, 0.20, 0.21, 0.19, 0.18, 0.30, 0.31],
    }
)
chain_dims = ["trade_date", "expiry", "strike"]

try:
    LabelledArray.from_dataframe(
        chain, cols=["vol"], dims=chain_dims, natural=["expiry", "strike"]
    )
except DuplicateIndexError as error:
    print(str(error)[:140], "...")

# %%
# Ranking distinct labels
# -----------------------
# The mapping form of ``natural`` says how each position is counted.
# ``"distinct"`` ranks the distinct labels, so the rows sharing an expiry share
# a position; ``"rows"`` is the numbering above, used here for the strikes
# beneath each expiry. Both axes are ragged and neither is padded by hand.

surface = LabelledArray.from_dataframe(
    chain,
    cols=["vol"],
    dims=chain_dims,
    natural={"expiry": "distinct", "strike": "rows"},
)
labelled_chain = LabelledArray.from_dataframe(chain, cols=["vol"], dims=chain_dims)

print(surface.shape, "against", labelled_chain.shape)
print(surface.coords["strike"])
assert surface.shape == (2, 2, 3)
assert labelled_chain.shape == (2, 3, 7)
assert surface.density > 3 * labelled_chain.density
assert_frame_equal(
    chain.sort(chain_dims), surface.to_dataframe(index=False).select(chain.columns)
)

# %%
# Coordinates that do not vary along every dimension
# --------------------------------------------------
# A natural index shrinks a *dimension*. The other half of the saving is a
# *coordinate* that does not vary along every dimension: give it at any shape
# that broadcasts to the array's, and it is stored once rather than repeated.
# An option chain is the case that hurts -- time to expiry is a property of the
# expiry, and every strike on that expiry shares it by definition.

expiries = np.array(["2024-03-15", "2024-06-21", "2024-09-20"], "datetime64[D]")
strikes = np.arange(4000, 4500, 5.0)
tau = (expiries - np.datetime64("2024-01-02")).astype(float) / 365.0

chain = LabelledArray(
    {"price": np.zeros((len(expiries), len(strikes)))},
    {"expiry": expiries, "strike": strikes},
    coords={"time_to_expiry": tau[:, None]},  # (3, 1) against (3, 100)
)

# Every consumer still reads a full-shape coordinate...
assert chain.coords["time_to_expiry"].shape == chain.shape
assert chain.coords["time_to_expiry"][1, 7] == tau[1]

# ...but the values behind it are held once.
held = chain.coords["time_to_expiry"]
print("read as :", held.shape, held.nbytes, "bytes")
print("held as :", held.base.shape, held.base.nbytes, "bytes")
assert held.base.nbytes == tau.nbytes

# %%
# The saving is at construction, and ends at the first slice
# ----------------------------------------------------------
# A broadcast coordinate is a zero-stride view, so it is read-only -- writing
# through one would update every cell sharing the element. It survives
# ``transpose``, ``copy(deep=False)`` and ``equals`` compactly, but the first
# ``select`` or ``reindex`` fancy-indexes it -- and a deep ``copy()``, the
# default, copies it -- and numpy materialises the full array there.

assert chain.coords["time_to_expiry"].flags.writeable is False
assert (
    chain.transpose("strike", "expiry").coords["time_to_expiry"].base.nbytes
    == tau.nbytes
)

sliced = chain.select({"strike": strikes[:10]})
assert sliced.coords["time_to_expiry"][2, 3] == tau[2], "values are intact"
print("after select:", sliced.coords["time_to_expiry"].nbytes, "bytes, materialised")

# %%
# Variables : broadcast at construction with ``broadcast=True``
# -------------------------------------------------------------
# The same saving is there for a *variable*, behind a flag. A vol surface over
# ``(date, expiry, strike)`` often carries a per-expiry quantity -- time to
# expiry here -- next to the per-strike vols. Give it as ``(date, expiry, 1)``
# and pass ``broadcast=True``: it is stored as a read-only zero-stride view of
# the full shape, so every consumer still reads one value per cell.
#
# The flag is stricter than numpy. Each array needs one axis per dimension,
# length 1 where it does not vary, so a ``(expiry,)`` array cannot silently land
# on the strike axis. ``dims`` must be given, since the shape can no longer be
# read off the first variable.

dates = np.array(["2024-01-02", "2024-01-03"], "datetime64[D]")
snap_tau = (expiries[None, :] - dates[:, None]).astype(float) / 365.0
vols = np.full((len(dates), len(expiries), len(strikes)), 0.2)

surface = LabelledArray(
    {"vol": vols, "time_to_expiry": snap_tau[:, :, None]},  # (2, 3, 1) vs (2, 3, 100)
    {"date": dates, "expiry": expiries, "strike": strikes},
    broadcast=True,
)

stored = surface["time_to_expiry"]
assert stored.shape == surface.shape and stored.strides[-1] == 0
assert not stored.flags.writeable
print("vol           :", surface["vol"].nbytes, "bytes")
print("time_to_expiry:", stored.base.nbytes, "bytes held, read as", stored.shape)

# The round trip to a DataFrame writes one row per cell, as for any variable.
frame = surface.to_dataframe(index=False)
assert frame.height == surface.size

# %%
# When sparsity is unavoidable
# ----------------------------
# ``density`` is the number to watch: it is the fraction of cells a row
# populates. A low density on a large array means the dense layout is the wrong
# one -- see TODO.md for where that is heading.

print(f"labelled: {labelled.density:.2%} of {labelled.size} cells")
print(f"natural : {natural.density:.2%} of {natural.size} cells")
