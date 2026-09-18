"""
Round trips through a DataFrame
===============================

Labelled arrays are usually reached from a DataFrame: some columns identify a
cell, the rest become variables. The conversion goes both ways: polars is the
default, and pandas works the same way when asked for by name.
"""

# %%
# A long-format frame
# -------------------
# One row per (region, product) pair, with two value columns. The pair
# ('south', 'BBB') has no row at all -- the frame is sparse.

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from labelled_array import LabelledArray

records = {
    "region": ["north", "north", "south", "north"],
    "product": ["AAA", "BBB", "AAA", "CCC"],
    "units": [10, 20, 30, 40],
    "price": [1.5, 2.5, 3.5, 4.5],
}
df = pl.DataFrame(records)
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
# The mask records which cells came from a row, so an integer column keeps its
# dtype: an empty cell holds a placeholder zero and the mask says it is absent.
# ``11_presence_mask.py`` covers the mask end to end.

print(sales["units"])
print(sales.mask)

assert sales.dtypes["units"] == np.dtype("int64")
assert sales["units"][1, 1] == 0
assert not sales.mask[1, 1]

# %%
# Converting back
# ---------------
# Only the populated cells become rows, so the round trip is exact -- up to row
# order, which follows the dimension labels. With no engine named the result
# is a polars frame, and the dimensions are ordinary columns.

out = sales.to_dataframe()
assert isinstance(out, pl.DataFrame)
assert out.columns == ["region", "product", "units", "price"]
assert_frame_equal(df.sort(["region", "product"]), out)
print(out)

# %%
# pandas: dimensions as an index
# ------------------------------
# A pandas frame with a named index needs no ``dims`` argument -- the index
# levels supply them. On the way out, ``to_dataframe("pandas")`` puts the
# dimensions back into a MultiIndex; ``index=False`` leaves them as columns.

import pandas as pd  # noqa: E402

indexed = pd.DataFrame(records).set_index(["region", "product"]).sort_index()
assert LabelledArray.from_dataframe(indexed).equals(sales)
pd.testing.assert_frame_equal(indexed, sales.to_dataframe("pandas"))
pd.testing.assert_frame_equal(
    indexed.reset_index(), sales.to_dataframe("pandas", index=False)
)
print(sales.to_dataframe("pandas"))

# %%
# polars and pandas side by side
# ------------------------------
# polars has no index, so its dimensions are always ordinary columns and
# ``dims`` is always passed. Either library builds the same array;
# ``13_polars_and_pandas.py`` sets the two side by side.

from_pandas = LabelledArray.from_dataframe(
    pd.DataFrame(records), cols=["units", "price"], dims=["region", "product"]
)
assert from_pandas.equals(sales)

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

duplicated = pl.DataFrame(
    {"region": ["north", "north"], "product": ["AAA", "AAA"], "units": [1, 2]}
)
try:
    LabelledArray.from_dataframe(duplicated, dims=["region", "product"])
except DuplicateIndexError as error:
    print(error)
