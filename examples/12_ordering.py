"""
What keeps its order, and what does not
=======================================

Two orders matter in a labelled array: the order of the *dimensions*, which is
the order of the axes, and the order of the *labels* within each dimension.
Selection changes only the second. This example walks through every operation
that touches either, so that neither ever moves by surprise.
"""

# %%
# The starting order
# ------------------
# ``from_dataframe`` sorts the labels of every dimension, so the layout does
# not depend on the order of the rows. The order of the axes is the order of
# ``dims=``, and nothing else.

import polars as pl

from labelled_array import DimensionError, LabelledArray

df = pl.DataFrame(
    {
        "region": ["south", "north", "south", "north", "south", "north"],
        "product": ["CCC", "BBB", "AAA", "CCC", "BBB", "AAA"],
        "sector": ["tech", "tech", "fin", "tech", "tech", "fin"],
        "units": [6, 2, 4, 3, 5, 1],
    }
)
sales = LabelledArray.from_dataframe(
    df, cols=["units"], dims=["region", "product"], alt={"sector": "product"}
)

print(sales["units"])
assert sales.dims.names == ("region", "product")
assert sales.dims["region"].tolist() == ["north", "south"], "sorted, not first-seen"
assert sales["units"].tolist() == [[1, 2, 3], [4, 5, 6]]

flipped = LabelledArray.from_dataframe(df, cols=["units"], dims=["product", "region"])
assert flipped.dims.names == ("product", "region")

# %%
# Selection never moves an axis
# -----------------------------
# The order in which dimensions are *named* in a selection means nothing. The
# axes of the result are the axes of the array, in the same order.

picked = sales.select({"product": ["AAA", "CCC"], "region": ["south"]})
assert picked.dims.names == ("region", "product")
assert picked.equals(sales.select({"region": ["south"], "product": ["AAA", "CCC"]}))

# %%
# Labels come back in the order asked for
# ---------------------------------------
# What a selection does reorder is the labels *within* the dimension selected.
# Asking for them backwards reverses that axis, by label or by position alike.

backwards = sales.select({"product": ["CCC", "BBB", "AAA"]})
print(backwards["units"])
assert backwards.dims["product"].tolist() == ["CCC", "BBB", "AAA"]
assert backwards["units"].tolist() == [[3, 2, 1], [6, 5, 4]]
assert backwards.equals(sales.isel(product=[2, 1, 0]))

# %%
# drop removes, it does not reorder
# ---------------------------------
# A scalar label leaves a length-one dimension behind, so the rank never
# depends on how many labels were asked for. ``drop=True`` removes the
# dimensions that were *named* and came out with one entry; the rest keep
# their order.

assert sales.select({"region": "north"}).shape == (1, 3)
assert sales.select({"region": "north"}, drop=True).dims.names == ("product",)

one_product = sales.select({"product": "BBB"})
assert one_product.select({"region": "north"}, drop=True).dims.names == ("product",)

# %%
# Selecting on an alternative dimension
# -------------------------------------
# An alternative dimension's labels repeat, so one label can keep several
# entries. They come back in the order the labels were requested, and in their
# existing order within each label.

by_sector = sales.select({"sector": ["tech", "fin"]})
assert by_sector.dims["product"].tolist() == ["BBB", "CCC", "AAA"]
assert by_sector.dims.alt["sector"].tolist() == ["tech", "tech", "fin"]

# %%
# Two selections along one axis are refused
# -----------------------------------------
# ``sector`` and ``product`` both select along the product axis. Their
# intersection has no order that either request defines, so the call is
# refused; chaining two calls says which order is wanted.

try:
    sales.select({"sector": "tech", "product": ["CCC", "BBB"]})
except DimensionError as error:
    print(error)

chained = sales.select({"sector": "tech"}).select({"product": ["CCC", "BBB"]})
assert chained.dims["product"].tolist() == ["CCC", "BBB"]

# %%
# transpose is the explicit axis move
# -----------------------------------
# It is the only method whose purpose is to reorder the axes. With no
# arguments it reverses them.

turned = sales.transpose("product", "region")
assert turned.dims.names == ("product", "region")
assert turned["units"].tolist() == [[1, 4], [2, 5], [3, 6]]
assert sales.transpose().equals(turned)

# %%
# Arithmetic follows the left operand
# -----------------------------------
# Operands are matched by dimension *name*, so their axis orders may differ.
# The right one is transposed to the left one's order, and only labels both
# sides have are kept -- in the left operand's order.

assert (sales + turned).dims.names == ("region", "product")
assert (turned + sales).dims.names == ("product", "region")
assert (sales + turned)["units"].tolist() == [[2, 4, 6], [8, 10, 12]]

assert (backwards + sales).dims["product"].tolist() == ["CCC", "BBB", "AAA"]

# %%
# align joins labels and leaves the axes alone
# --------------------------------------------
# Alignment is about labels, not axis order, so nothing is transposed. An
# inner join keeps the first array's order; an outer join sorts the union --
# unless the labels already agree, in which case there is nothing to join and
# they are left exactly as they are.

left, right = backwards.align(turned)
assert left.dims.names == ("region", "product")
assert right.dims.names == ("product", "region")
assert right.dims["product"].tolist() == ["CCC", "BBB", "AAA"], "the first's order"

two = sales.select({"product": ["CCC", "AAA"]})
left, right = two.align(sales, join="outer")
assert left.dims["product"].tolist() == ["AAA", "BBB", "CCC"], "a sorted union"

left, right = backwards.align(backwards, join="outer")
assert left.dims["product"].tolist() == ["CCC", "BBB", "AAA"], "nothing to join"

# %%
# concat takes its axis order from the first array
# ------------------------------------------------
# Every operand is transposed to the first one's order. Along an existing
# dimension the labels are appended as given, not sorted, and a *new*
# dimension becomes the first axis.

north, south = (sales.select({"region": [name]}) for name in ("north", "south"))
joined = LabelledArray.concat([south, north.transpose()], dim="region")
assert joined.dims.names == ("region", "product")
assert joined.dims["region"].tolist() == ["south", "north"], "appended, not sorted"

years = LabelledArray.concat([sales, sales * 2], dim="year", labels=[2023, 2024])
assert years.dims.names == ("year", "region", "product")

# %%
# reindex and groupby
# -------------------
# ``reindex`` gives back exactly the labels asked for, in that order. A
# grouped reduction keeps the dimension's place among the axes but relabels
# it with the group values, which are sorted like any other labels.

assert sales.reindex({"product": ["CCC", "AAA"]}).dims["product"].tolist() == [
    "CCC",
    "AAA",
]

per_sector = sales.groupby("sector").sum()
print(per_sector["units"])
assert per_sector.dims.names == ("region", "product")
assert per_sector.dims["product"].tolist() == ["fin", "tech"]
assert per_sector["units"].tolist() == [[1, 5], [4, 11]]

# %%
# Reductions drop axes and keep the rest in order
# -----------------------------------------------

cube = years.sum("region")
assert cube.dims.names == ("year", "product")

# %%
# Out to a frame, and back
# ------------------------
# Columns are written dimensions first, then coordinates, then alternative
# dimensions, then variables. Rows follow the array's layout, so after a
# reordering selection they are no longer sorted -- and reading the frame back
# sorts the labels again.

out = backwards.to_dataframe()
print(out)
assert out.columns == ["region", "product", "sector", "units"]
assert out["product"].to_list() == ["CCC", "BBB", "AAA"] * 2

back = LabelledArray.from_dataframe(
    out, cols=["units"], dims=["region", "product"], alt={"sector": "product"}
)
assert back.dims["product"].tolist() == ["AAA", "BBB", "CCC"]
assert back.equals(sales)

# %%
# A natural index takes its dimension's place
# -------------------------------------------
# ``natural=`` replaces a dimension with positions, renamed ``<name>_idx`` on
# the same axis, and keeps the values as a coordinate. ``to_dataframe`` writes
# those values back into the column the dimension came from.

positional = LabelledArray.from_dataframe(
    df, cols=["units"], dims=["region", "product"], natural=["product"]
)
assert positional.dims.names == ("region", "product_idx")
assert positional.to_dataframe().columns == ["region", "product", "units"]
