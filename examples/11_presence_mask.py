"""
The presence mask, end to end
=============================

A dense array has a cell for every combination of labels, whether or not the
data had a row for it. The mask is how a ``LabelledArray`` remembers which cells
are real. This example follows it from the frame it comes from, through every
operation that changes it, and back out to a frame.
"""

# %%
# Where a mask comes from
# -----------------------
# Four rows over a two by three layout leave two cells with nothing behind
# them. ``mask`` is True where a row existed -- it records *presence*, the
# opposite sense to ``numpy.ma`` -- and ``density`` is the populated share.

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from labelled_array import LabelledArray

df = pl.DataFrame(
    {
        "region": ["north", "north", "south", "north"],
        "product": ["AAA", "BBB", "AAA", "CCC"],
        "units": [10, 20, 30, 40],
        "price": [1.5, 2.5, 3.5, 4.5],
    }
)
sales = LabelledArray.from_dataframe(df, dims=["region", "product"])

print(sales.mask)
assert sales.mask.tolist() == [[True, True, True], [True, False, False]]
assert sales.density == 4 / 6

# %%
# It is always numpy, and read-only
# ---------------------------------
# The variables may live on JAX or torch; the mask never does. It is frozen so
# that it cannot drift out of step with the variables it describes.

assert isinstance(sales.mask, np.ndarray) and sales.mask.dtype == bool
try:
    sales.mask[1, 1] = True
except ValueError as error:
    print(error)

# %%
# A dense frame keeps no mask
# ---------------------------
# ``None`` means every cell is populated. One row per cell is detected from the
# row count alone, so a dense frame never pays for a mask.

north = LabelledArray.from_dataframe(
    df.filter(pl.col("region") == "north"), dims=["region", "product"]
)
assert north.mask is None
assert north.density == 1.0

# %%
# Why a mask and not NaN
# ----------------------
# NaN only exists for floats, so marking a gap with it would widen every other
# dtype. The gap instead holds a placeholder the dtype can represent, and the
# mask -- not the value -- is what says the cell is empty.

assert sales.dtypes["units"] == np.dtype("int64"), "not widened to float"
assert sales["units"][1, 1] == 0, "a placeholder, not a sale of nothing"
assert np.isnan(sales["price"][1, 1]), "floats get NaN, but the mask still rules"
assert not sales.mask[1, 1]

# %%
# Only present cells become rows
# ------------------------------
# ``to_dataframe`` writes one row per masked-in cell, which is what makes the
# round trip exact for integers, booleans and strings as much as for floats.

assert_frame_equal(df.sort(["region", "product"]), sales.to_dataframe())

# %%
# Opting out loses information
# ----------------------------
# ``mask=False`` saves the memory, and with it goes the knowledge of which
# cells were real: six rows come back from four. ``dropna`` drops a row only
# when *every* variable in it is null, so it recovers the four rows for floats
# alone -- an integer placeholder is a zero like any other.

unmasked = LabelledArray.from_dataframe(df, dims=["region", "product"], mask=False)
assert unmasked.mask is None
assert len(unmasked.to_dataframe()) == 6
assert len(unmasked.to_dataframe(dropna=True)) == 6, "units is never null"
assert len(unmasked[["price"]].to_dataframe(dropna=True)) == 4

# %%
# Arithmetic keeps what is present in both
# ----------------------------------------
# A result cell is real only if both operands had one, so the masks combine
# with ``&``. An operand with no mask counts as fully present, and a scalar
# leaves the mask alone.

promo = LabelledArray(
    {"units": np.ones((2, 3), dtype="int64")},
    {"region": ["north", "south"], "product": ["AAA", "BBB", "CCC"]},
    mask=[[True, False, True], [True, True, True]],
)
both = sales[["units"]] + promo
print(both.mask)
assert both.mask.tolist() == (sales.mask & promo.mask).tolist()
assert (sales * 2).mask.tolist() == sales.mask.tolist()

# %%
# Operations that invent cells mark them absent
# ---------------------------------------------
# Reindexing onto a label the array never had, or aligning with an outer join,
# creates cells that no row stands behind. They arrive masked out.

wider = sales.reindex({"region": ["north", "south", "west"]})
assert wider.mask.tolist()[2] == [False, False, False]
assert len(wider.to_dataframe()) == 4, "the invented cells are not rows"

east = LabelledArray(
    {"units": np.array([[7, 8, 9]])},
    {"region": ["east"], "product": ["AAA", "BBB", "CCC"]},
)
assert east.mask is None
_, aligned = sales[["units"]].align(east, join="outer")
assert aligned.dims["region"].tolist() == ["east", "north", "south"]
assert aligned.mask.tolist()[1] == [False, False, False]

# %%
# Concatenation treats a maskless operand as present
# --------------------------------------------------

stacked = LabelledArray.concat([sales[["units"]], east], dim="region")
print(stacked.mask)
assert stacked.mask.tolist() == [*sales.mask.tolist(), [True, True, True]]

# %%
# Reductions skip absent cells, not NaN
# -------------------------------------
# An absent cell contributes nothing: a mean divides by the populated count,
# and a sum over nothing at all is *absent*, not zero.

mean_price = wider.mean("product")["price"]
assert mean_price[0] == (1.5 + 2.5 + 4.5) / 3
assert mean_price[1] == 3.5, "south has one price, so the mean divides by one"

totals = wider.sum("product")
print(totals.to_dataframe())
assert totals.mask.tolist() == [True, True, False], "west sold nothing we know of"

# %%
# A NaN that is present is data
# -----------------------------
# There is no ``skipna``. A NaN in a masked-in cell was in the source, so it
# propagates as it would in numpy; only the mask takes a cell out of a result.

observed = LabelledArray({"w": np.array([1.0, np.nan, 3.0])}, {"k": ["a", "b", "c"]})
assert np.isnan(observed.sum()["w"])
hidden = LabelledArray(
    {"w": np.array([1.0, np.nan, 3.0])},
    {"k": ["a", "b", "c"]},
    mask=[True, False, True],
)
assert hidden.sum()["w"] == 4.0

# %%
# Forward filling makes cells present
# -----------------------------------
# ``ffill`` carries the last *present* cell forward over the mask -- not over
# NaN -- and the cells it fills become present, or ``to_dataframe`` would drop
# the very rows the fill was for. ``limit`` counts positions, and a gap before
# the first observation stays a gap. ``09_ffill_and_staleness.py`` takes this
# further.

quotes = LabelledArray(
    {"px": np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 9.0]])},
    {"asset": ["AAA"], "day": np.arange(6)},
    mask=[[False, True, False, False, False, True]],
)
filled = quotes.ffill("day")
assert filled["px"].tolist() == [[0.0, 5.0, 5.0, 5.0, 5.0, 9.0]]
assert filled.mask.tolist() == [[False, True, True, True, True, True]]

capped = quotes.ffill("day", limit=1)
assert capped.mask.tolist() == [[False, True, True, False, False, True]]

# %%
# Filling with a value
# --------------------
# ``fillna`` with a scalar gives every absent cell a value, so nothing is
# absent any more and the mask is dropped. Filling only some variables leaves
# the others with gaps, and the mask stays.

assert sales.fillna(0).mask is None
assert len(sales.fillna(0).to_dataframe()) == 6
assert sales.fillna({"units": 0}).mask is not None

# %%
# Two things to watch for
# -----------------------
# A rolling window that needs only one observation is satisfied wherever any
# cell in it is present, and when that is everywhere the mask goes altogether.
# And ``mask=None`` is not ``equals`` to an all-True mask, though the two write
# identical frames.

window = quotes.isel(day=[1, 2, 3])
assert window.mask.tolist() == [[True, False, False]]
rolled = window.rolling("day", 3, min_periods=1).sum()
assert rolled.mask is None, "every trailing window reaches back to the 5.0"
assert len(rolled.to_dataframe()) == 3, "one row in, three rows out"

all_true = LabelledArray({"w": np.zeros(2)}, {"k": [1, 2]}, mask=[True, True])
no_mask = LabelledArray({"w": np.zeros(2)}, {"k": [1, 2]})
assert not all_true.equals(no_mask)
assert_frame_equal(all_true.to_dataframe(), no_mask.to_dataframe())

# %%
# A different kind of mask
# ------------------------
# ``isel`` also takes a boolean array, which picks *positions* along one
# dimension. It shares a word with the presence mask and nothing else.

picked = sales.isel(product=[True, False, True])
assert picked.dims["product"].tolist() == ["AAA", "CCC"]
assert picked.mask.tolist() == [[True, True], [True, False]]

# %%
# Where else the mask appears
# ---------------------------
# ``02_from_dataframe.py`` introduces it, ``04_natural_index_and_sparsity.py``
# shows a natural index keeping density high, ``08_combining_and_reducing.py``
# uses the rules above on a real panel, and ``07_timings_at_scale.py`` measures
# what the mask costs.
