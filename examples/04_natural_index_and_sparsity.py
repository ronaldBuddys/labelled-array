"""
Natural indices and sparsity
============================

A dense array spans the product of its dimension labels, which is wasteful when
one dimension's labels are nearly unique per group -- dates, most of the time.
Replacing that dimension with a natural index collapses the array while keeping
the original values, so the round trip is still exact.
"""

# %%
# The problem: a date per row
# ---------------------------
# Three groups of four rows, each with dates that appear in no other group.

import numpy as np
import pandas as pd

from labelled_array import LabelledArray

panel = pd.DataFrame(
    {
        "group": np.repeat(["A", "B", "C"], 4),
        "item": list("wxyz") * 3,
        "date": np.arange(12).astype("datetime64[D]"),
        "value": np.arange(12, dtype=float),
    }
)
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
# ``to_dataframe`` writes the coordinate back out under its original name.

out = natural.to_dataframe(index=False)
pd.testing.assert_frame_equal(
    panel.sort_values(["group", "item", "date"]).reset_index(drop=True), out
)
print(out.head())

# %%
# ...or the index itself, if that is what you want
# ------------------------------------------------

positions = natural.to_dataframe(index=False, expand_natural=False)
assert list(positions.columns) == ["group", "item", "date_idx", "date", "value"]
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

ragged = pd.DataFrame(
    {
        "group": ["A", "A", "A", "B"],
        "date": np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"], "datetime64[D]"
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
pd.testing.assert_frame_equal(
    ragged.sort_values(["group", "date"]).reset_index(drop=True),
    uneven.to_dataframe(index=False),
)

# %%
# When sparsity is unavoidable
# ----------------------------
# ``density`` is the number to watch: it is the fraction of cells a row
# populates. A low density on a large array means the dense layout is the wrong
# one -- see TODO.md for where that is heading.

print(f"labelled: {labelled.density:.2%} of {labelled.size} cells")
print(f"natural : {natural.density:.2%} of {natural.size} cells")
