"""
.. _polars_and_pandas_example:

``from_dataframe``, ``to_dataframe`` : one API, two DataFrame engines
=====================================================================

``from_dataframe`` takes a polars frame or a pandas one, and ``to_dataframe``
writes either. The array in between is the same whichever library it came from.
This example runs the two side by side, and then shows the few places where the
libraries themselves differ and the difference shows through.
"""

# %%
# The same records in both libraries
# ----------------------------------
# polars is the default engine. pandas is just as welcome: nothing below needs
# a different method, only a different frame.

import numpy as np
import pandas as pd
import polars as pl

from labelled_array import DimensionError, LabelledArray

records = {
    "region": ["north", "north", "south", "north"],
    "product": ["AAA", "BBB", "AAA", "CCC"],
    "units": [10, 20, 30, 40],
    "price": [1.5, 2.5, 3.5, 4.5],
}
polars_frame = pl.DataFrame(records)
pandas_frame = pd.DataFrame(records)

# %%
# The same array
# --------------
# The engine is detected from the frame. Labels, their order, their dtype, the
# variables and the mask all agree, so the library a frame came from never
# shows in the array built from it.

from_polars = LabelledArray.from_dataframe(polars_frame, dims=["region", "product"])
from_pandas = LabelledArray.from_dataframe(pandas_frame, dims=["region", "product"])

assert from_polars.equals(from_pandas)
assert from_polars.dtypes == from_pandas.dtypes
assert from_polars.mask.tolist() == from_pandas.mask.tolist()
for name in from_polars.dims.names:
    assert from_polars.dims[name].dtype == from_pandas.dims[name].dtype == object

# %%
# How that is kept true
# ---------------------
# Both adapters hand numpy arrays to one shared conversion. The one exception
# is factorizing a string dimension, which polars does natively because it is
# several times faster -- and which must reproduce the numpy reference exactly,
# or fall back to it. The test suite checks the one against the other.

from labelled_array import _reshape  # noqa: E402
from labelled_array.adapters import get_adapter  # noqa: E402

reference = _reshape.factorize(np.array(records["region"], dtype=object))
for engine, frame in (("polars", polars_frame), ("pandas", pandas_frame)):
    labels, codes = get_adapter(engine).factorize(frame, "region")
    assert labels.tolist() == reference[0].tolist() == ["north", "south"]
    assert codes.tolist() == reference[1].tolist()

# %%
# Writing out: polars unless asked
# --------------------------------
# ``to_dataframe()`` returns polars. Pass ``"pandas"`` for a pandas frame; the
# content is the same, column for column.

as_polars = from_polars.to_dataframe()
as_pandas = from_polars.to_dataframe("pandas", index=False)

assert isinstance(as_polars, pl.DataFrame)
assert isinstance(as_pandas, pd.DataFrame)
assert as_polars.columns == list(as_pandas.columns)
for name in as_polars.columns:
    assert as_polars[name].to_list() == as_pandas[name].tolist()
print(as_polars)

# %%
# The source engine is not remembered
# -----------------------------------
# An array holds numpy arrays and no memory of a DataFrame library, so the
# default output does not depend on where the data came from. An array built
# from pandas still writes polars; say ``"pandas"`` to go back.

assert isinstance(from_pandas.to_dataframe(), pl.DataFrame)
assert isinstance(from_pandas.to_dataframe(engine="pandas"), pd.DataFrame)

# %%
# Where they differ: the index
# ----------------------------
# pandas has an index and polars does not. A pandas frame's named index levels
# supply ``dims`` when none are given, and ``to_dataframe("pandas")`` puts the
# dimensions back into one, so an indexed frame round trips as it is.

indexed = pandas_frame.set_index(["region", "product"]).sort_index()
from_index = LabelledArray.from_dataframe(indexed)
assert from_index.equals(from_polars)
pd.testing.assert_frame_equal(indexed, from_index.to_dataframe("pandas"))
print(from_index.to_dataframe("pandas"))

# %%
# ...which polars frames never have
# ---------------------------------
# With nothing to take the dimensions from, a polars frame has to be told.
# ``index=`` is accepted and ignored, so code can switch engines freely.

try:
    LabelledArray.from_dataframe(polars_frame)
except DimensionError as error:
    print(error)

assert from_polars.to_dataframe(index=True).equals(from_polars.to_dataframe())

# %%
# Where they differ: dtypes
# -------------------------
# Values pass through numpy, so each library gets back whatever it makes of a
# numpy dtype. A calendar date is the clearest case: polars has a ``Date`` and
# pandas widens it to a timestamp.

days = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
daily = LabelledArray({"w": np.array([1.0, 2.0])}, {"day": days})

assert daily.to_dataframe().schema["day"] == pl.Date
assert daily.to_dataframe("pandas", index=False)["day"].dtype.kind == "M"
print(daily.to_dataframe().schema, daily.to_dataframe("pandas").index.dtype)

# %%
# Units one library lacks are widened exactly
# -------------------------------------------
# Subtracting two dates gives ``timedelta64[D]``, which polars has no dtype
# for. It is written in milliseconds rather than refused: the unit changes and
# the values do not. The README tabulates every such case for both libraries.

age = LabelledArray({"age": days - days[0]}, {"day": days})
assert age.dtypes["age"] == np.dtype("timedelta64[D]")
assert age.to_dataframe().schema["age"] == pl.Duration("ms")
assert (
    age.to_dataframe()["age"].to_numpy().astype("timedelta64[D]").tolist()
    == (days - days[0]).tolist()
)

# %%
# Crossing over
# -------------
# Because the array is the same, a frame can go in from one library and come
# out of the other, in either direction, and nothing is lost on the way.

via_polars = LabelledArray.from_dataframe(
    from_pandas.to_dataframe("polars"), dims=["region", "product"]
)
via_pandas = LabelledArray.from_dataframe(from_polars.to_dataframe("pandas"))
assert via_polars.equals(from_pandas)
assert via_pandas.equals(from_polars)

# %%
# Forcing an engine, or adding one
# --------------------------------
# ``engine=`` overrides detection on the way in. Engines live in a registry, so
# another DataFrame library is one ``DataFrameAdapter`` subclass and a call to
# ``register_adapter`` away.

from labelled_array.adapters import available_adapters  # noqa: E402

forced = LabelledArray.from_dataframe(
    polars_frame, dims=["region", "product"], engine="polars"
)
assert forced.equals(from_polars)
print("engines:", available_adapters())
