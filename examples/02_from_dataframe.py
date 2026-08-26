"""
Round trips with pandas and polars
==================================

Labelled arrays are usually reached from a DataFrame: some columns identify a
cell, the rest become variables. The conversion goes both ways, and both
DataFrame libraries give exactly the same result.
"""

# %%
# A long-format frame
# -------------------
# One row per (region, product) pair, with two value columns. The pair
# ('south', 'BBB') has no row at all -- the frame is sparse.

import numpy as np
import pandas as pd
import polars as pl

from labelled_array import LabelledArray

records = {
    "region": ["north", "north", "south", "north"],
    "product": ["AAA", "BBB", "AAA", "CCC"],
    "units": [10, 20, 30, 40],
    "price": [1.5, 2.5, 3.5, 4.5],
}
df = pd.DataFrame(records)
print(df)

# %%
# Converting to an array
# ----------------------
# Each dimension column contributes its sorted unique values as labels, so the
# array is 2 x 3 even though only four of those six cells exist.

sales = LabelledArray.from_dataframe(
    df, cols=["units", "price"], dims=["region", "product"]
)
print(sales)

assert sales.shape == (2, 3)
assert sales.dims["product"].tolist() == ["AAA", "BBB", "CCC"]
assert sales.density == 4 / 6

# %%
# The presence mask
# -----------------
# The mask records which cells came from a row. It is what lets an integer
# column keep its dtype: the two empty cells hold a placeholder zero, and the
# mask -- not the value -- is what says they are absent.

print(sales["units"])
print(sales.mask)

assert sales.dtypes["units"] == np.dtype("int64")
assert sales["units"][1, 1] == 0
assert not sales.mask[1, 1]

# %%
# Converting back
# ---------------
# Only the populated cells become rows, so the round trip is exact -- up to row
# order, which follows the dimension labels.

out = sales.to_dataframe(index=False)
pd.testing.assert_frame_equal(
    df.sort_values(["region", "product"]).reset_index(drop=True), out
)
print(out)

# %%
# Keeping the dimensions as an index
# ----------------------------------
# By default pandas gets the dimension columns back as a MultiIndex, and a
# frame that already has one needs no ``dims`` argument at all.

indexed = df.set_index(["region", "product"]).sort_index()
assert LabelledArray.from_dataframe(indexed).equals(sales)
pd.testing.assert_frame_equal(indexed, sales.to_dataframe())

# %%
# polars gives the same answer
# ----------------------------
# polars has no index, so dimensions are always ordinary columns. The array
# built from either library is the same object.

polars_frame = pl.DataFrame(records)
from_polars = LabelledArray.from_dataframe(
    polars_frame, cols=["units", "price"], dims=["region", "product"]
)

assert from_polars.equals(sales)
print(sales.to_dataframe("polars"))

# %%
# Choosing the fill and the dtype
# -------------------------------
# The placeholder and the dtype can be set per variable when the defaults do
# not suit.

filled = LabelledArray.from_dataframe(
    df,
    cols=["units", "price"],
    dims=["region", "product"],
    fill_value={"units": -1},
    dtype={"price": "float32"},
)
assert filled["units"][1, 1] == -1
assert filled.dtypes["price"] == np.dtype("float32")

# %%
# Rows must identify a cell
# -------------------------
# Two rows for the same cell would be ambiguous, so the conversion refuses it
# and names the offending labels.

from labelled_array import DuplicateIndexError  # noqa: E402

duplicated = pd.DataFrame(
    {"region": ["north", "north"], "product": ["AAA", "AAA"], "units": [1, 2]}
)
try:
    LabelledArray.from_dataframe(duplicated, dims=["region", "product"])
except DuplicateIndexError as error:
    print(error)
