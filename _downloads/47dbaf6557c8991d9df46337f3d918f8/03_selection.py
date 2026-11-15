"""
.. _selection_example:

``select`` : selecting by label
===============================

Selection is by label, not position, and the shape of the result never depends
on how many labels were asked for: a single label still yields a dimension of
length one unless it is explicitly dropped.
"""

# %%
# The array to work on
# --------------------

import numpy as np

from labelled_array import LabelledArray, LabelNotFoundError

sales = LabelledArray(
    {"units": np.arange(12).reshape(2, 3, 2)},
    dims={
        "region": ["north", "south"],
        "product": ["AAA", "BBB", "CCC"],
        "year": [2023, 2024],
    },
)
print(sales)

# %%
# One label per dimension
# -----------------------
# The dimension survives with a single label in it, so the rank of the result
# is predictable.

one_region = sales.select({"region": "north"})
assert one_region.shape == (1, 3, 2)
assert one_region.dims["region"].tolist() == ["north"]

# %%
# Dropping the dimension
# ----------------------
# ``drop=True`` removes the dimensions the selection reduced to one entry, and
# only those.

dropped = sales.select({"region": "north"}, drop=True)
assert dropped.dims.names == ("product", "year")
assert dropped.shape == (3, 2)

# %%
# Several dimensions at once, and the requested order
# ---------------------------------------------------
# Labels come back in the order they were asked for, so a selection can also
# reorder a dimension.

subset = sales.select({"region": "south", "product": ["CCC", "AAA"]})
assert subset.shape == (1, 2, 2)
assert subset.dims["product"].tolist() == ["CCC", "AAA"]

reversed_products = sales.select(product=["CCC", "BBB", "AAA"])
assert reversed_products.shape == sales.shape
np.testing.assert_array_equal(reversed_products["units"], sales["units"][:, ::-1])

# %%
# Labels must exist
# -----------------
# By default a label that is not there is an error naming the dimension and the
# missing labels, rather than a silently smaller result.

try:
    sales.select({"product": ["AAA", "ZZZ"]})
except LabelNotFoundError as error:
    print(error)

lenient = sales.select({"product": ["AAA", "ZZZ"]}, missing="ignore")
assert lenient.dims["product"].tolist() == ["AAA"]

# %%
# Equivalent spellings
# --------------------
# Keyword arguments, a mapping, and indexing with a mapping all do the same
# thing; chaining selections matches doing them at once.

assert sales.select(region="north").equals(sales.select({"region": "north"}))
assert sales[{"region": "north"}].equals(sales.select({"region": "north"}))
assert (
    sales.select({"region": "north"})
    .select({"year": 2024})
    .equals(sales.select({"region": "north", "year": 2024}))
)

# %%
# Selecting by position
# ---------------------
# ``isel`` is the positional counterpart, and takes slices and boolean masks.

assert sales.isel(product=slice(0, 2)).shape == (2, 2, 2)
assert sales.isel(product=[True, False, True]).dims["product"].tolist() == [
    "AAA",
    "CCC",
]
assert sales.isel(year=-1).dims["year"].tolist() == [2024]
assert sales.select({"product": ["CCC", "AAA"]}).equals(sales.isel({"product": [2, 0]}))

# %%
# A selection is still a labelled array
# -------------------------------------
# The mask, the coordinates and the attributes all come along, so a selection
# converts back to a DataFrame like any other array.

print(sales.select({"region": "north"}, drop=True).to_dataframe().head())
