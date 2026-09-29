"""
Alternative dimensions
======================

A descriptive column often varies along one dimension and only that one: a
sector per asset, a currency per book, a maturity bucket per tenor. Making it a
dimension of its own spans every combination and leaves the array mostly empty.
An *alternative dimension* labels the entries of a dimension the array already
has, is stored once per entry rather than once per cell, and -- unlike a
coordinate -- can be selected on.
"""

# %%
# The column that does not want a dimension
# -----------------------------------------
# Four assets over three dates, each asset in a sector. Sector is a property of
# the asset, so as a third dimension it spans 4 x 3 x 2 cells to hold the 12
# that exist.

import numpy as np
import polars as pl

from labelled_array import DimensionError, LabelledArray, LabelNotFoundError

ASSETS = ["AAA", "BBB", "CCC", "DDD"]
DATES = ["d1", "d2", "d3"]
SECTORS = ["tech", "tech", "fin", "tech"]

panel = pl.DataFrame(
    {
        "asset": [asset for asset in ASSETS for _ in DATES],
        "date": DATES * len(ASSETS),
        "sector": [sector for sector in SECTORS for _ in DATES],
        "px": np.arange(12.0),
    }
)

spread = LabelledArray.from_dataframe(
    panel, cols=["px"], dims=["asset", "date", "sector"]
)
assert spread.shape == (4, 3, 2)
assert spread.density == 0.5

# %%
# Declaring one instead
# ---------------------
# ``alt_dims`` ties the column to the dimension it describes. The array keeps
# its two axes and is fully populated, and the four sector labels are held once
# each rather than at every cell.

book = LabelledArray.from_dataframe(
    panel, cols=["px"], dims=["asset", "date"], alt={"sector": "asset"}
)
assert book.shape == (4, 3)
assert book.density == 1.0
assert book.dims.alt["sector"].tolist() == SECTORS
assert book.dims.alt_ref("sector") == "asset"
print(book.dims)

# %%
# The same thing, given directly
# ------------------------------
# Outside a DataFrame it is one entry per label of the reference dimension,
# spelled either way.

direct = LabelledArray(
    {"px": np.arange(12.0).reshape(4, 3)},
    dims={"asset": ASSETS, "date": DATES},
    alt_dims={"sector": {"ref_dim": "asset", "labels": SECTORS}},
)
terse = LabelledArray(
    {"px": np.arange(12.0).reshape(4, 3)},
    dims={"asset": ASSETS, "date": DATES},
    alt_dims={"sector": ("asset", SECTORS)},
)
assert direct.equals(terse)
# Same array, bar the label dtype: a frame's string columns arrive as ``object``,
# which ``README.md`` tabulates under the round trip.
np.testing.assert_array_equal(direct["px"], book["px"])
assert direct.dims.alt["sector"].tolist() == book.dims.alt["sector"].tolist()

# %%
# Selecting on it
# ---------------
# A name resolves as a dimension first and an alternative dimension second, so
# ``select`` needs no new argument and the two kinds mix in one call. Because
# alternative labels repeat, one of them keeps several entries.

tech = book.select({"sector": "tech"})
assert tech.dims["asset"].tolist() == ["AAA", "BBB", "DDD"]
assert tech.shape == (3, 3)

one_cell = book.select({"sector": "fin", "date": "d2"})
assert one_cell.shape == (1, 1)
assert one_cell["px"].item() == 7.0

# %%
# Order, and dropping
# -------------------
# Labels come back in the order they were asked for, ascending within each
# label, and ``drop=True`` removes a dimension the selection reduced to one
# entry -- taking the alternative dimension with it.

reordered = book.select(sector=["fin", "tech"])
assert reordered.dims["asset"].tolist() == ["CCC", "AAA", "BBB", "DDD"]
assert reordered.dims.alt["sector"].tolist() == ["fin", "tech", "tech", "tech"]

dropped = book.select({"sector": "fin"}, drop=True)
assert dropped.dims.names == ("date",)
assert dropped.dims.alt == {}

# %%
# What it is not
# --------------
# It is not an axis, so ``isel`` cannot name it: it has no positions of its own,
# only the ones it points at. And two names selecting along one dimension is an
# intersection whose order neither request defines, so it is refused and asked
# for as two calls.

assert "sector" not in book.dims
assert book.shape == (4, 3)

try:
    book.isel(sector=0)
except DimensionError as error:
    print(error)

try:
    book.select({"sector": "tech", "asset": "AAA"})
except DimensionError as error:
    print(error)

both = book.select(sector="tech").select(asset=["AAA", "DDD"])
assert both.dims["asset"].tolist() == ["AAA", "DDD"]

# %%
# An absent label
# ---------------
# Absence is reported the same way a dimension's own labels report it.

try:
    book.select(sector="energy")
except LabelNotFoundError as error:
    print(error)

assert book.select({"sector": ["energy", "fin"]}, missing="ignore").dims[
    "asset"
].tolist() == ["CCC"]

# %%
# Through the operations
# ----------------------
# An alternative dimension describes the entries of its dimension, so it is
# carried by anything that keeps those entries and dropped by anything that
# replaces them with something else.

assert book.isel(asset=[3, 0]).dims.alt["sector"].tolist() == ["tech", "tech"]
assert book.transpose("date", "asset").dims.alt["sector"].tolist() == SECTORS
assert book.rename(asset="ticker").dims.alt_ref("sector") == "ticker"
assert book.reindex(asset=["DDD", "CCC"]).dims.alt["sector"].tolist() == ["tech", "fin"]
assert book.sum("date").dims.alt["sector"].tolist() == SECTORS

# A label the array never had has no description, and a null one could never be
# selected -- so the alternative dimension goes rather than being invented.
assert book.reindex(asset=["AAA", "ZZZ"]).dims.alt == {}

# The entries are gone entirely here, so there is nothing left to describe.
assert book.sum("asset").dims.alt == {}
assert book.groupby(np.array(SECTORS), dim="asset").mean().dims.alt == {}

# %%
# Grouping by one
# ---------------
# An alternative dimension is already one value per label of the dimension it
# describes, which is exactly what a grouping is -- so ``groupby`` takes its
# name and needs no ``dim=``. The grouped dimension comes back labelled by group
# value, so the alternative dimension it described is gone from the result.

by_sector = book.groupby("sector").mean()
assert by_sector.dims["asset"].tolist() == ["fin", "tech"]
assert by_sector.dims.alt == {}
np.testing.assert_array_equal(by_sector["px"][0], book["px"][2])

# The same thing spelled out, for a value that is not stored on the array.
spelled_out = book.groupby(np.array(SECTORS), dim="asset").mean()
np.testing.assert_array_equal(by_sector["px"], spelled_out["px"])
assert by_sector.dims["asset"].tolist() == spelled_out.dims["asset"].tolist()

# %%
# Joining two of them
# -------------------
# ``concat`` along the described dimension joins the descriptions with it. The
# operands must agree about any alternative dimension on a *different*
# dimension, since alignment puts them on one label set but not on one account
# of what it means.

more = LabelledArray(
    {"px": np.zeros((2, 3))},
    dims={"asset": ["EEE", "FFF"], "date": DATES},
    alt_dims={"sector": ("asset", ["energy", "fin"])},
)
joined = LabelledArray.concat([book, more], dim="asset")
assert joined.dims["asset"].tolist() == [*ASSETS, "EEE", "FFF"]
assert joined.dims.alt["sector"].tolist() == [*SECTORS, "energy", "fin"]
assert joined.select(sector="fin").dims["asset"].tolist() == ["CCC", "FFF"]

# %%
# The round trip
# --------------
# ``to_dataframe`` writes the alternative dimension out as an ordinary column,
# gathered at each cell's position along the dimension it describes, so the
# column that was folded in comes back out.

out = book.to_dataframe().sort(["asset", "date"])
assert out.columns == ["asset", "date", "sector", "px"]
assert set(zip(out["asset"], out["sector"], strict=True)) == set(
    zip(ASSETS, SECTORS, strict=True)
)
with pl.Config(tbl_rows=len(out)):
    print(out)

# %%
# Against a coordinate
# --------------------
# A coordinate labels every *cell* and is stored at the array's shape, so it can
# say something different in each one -- and it can be grouped by, but not
# selected on. An alternative dimension labels the *entries of one dimension*,
# is stored once per entry, and can be selected on. Reach for a coordinate when
# the value genuinely varies cell by cell, and for an alternative dimension when
# it is a property of one dimension's labels. See
# ``examples/04_natural_index_and_sparsity.py`` for the third case: a dimension
# whose labels are near-unique per group, which wants a natural index instead.

with_coord = LabelledArray(
    {"px": np.arange(12.0).reshape(4, 3)},
    dims={"asset": ASSETS, "date": DATES},
    coords={"traded": np.ones((4, 3), dtype=bool)},
)
assert with_coord.coords["traded"].shape == (4, 3)
assert book.dims.alt["sector"].shape == (4,)
