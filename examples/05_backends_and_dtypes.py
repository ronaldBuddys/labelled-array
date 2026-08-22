"""
Array backends and dtypes
=========================

The variables can live on numpy, JAX or torch, and the same conversion code
serves all of them. Dimension labels, the mask and the coordinates always stay
on numpy, because they are metadata rather than payload.
"""

# %%
# Moving between backends
# -----------------------

import numpy as np
import pandas as pd

from labelled_array import BackendDtypeError, LabelledArray
from labelled_array.backends import available_backends

print("available backends:", available_backends())

df = pd.DataFrame(
    {
        "region": ["north", "north", "south"],
        "product": ["AAA", "BBB", "AAA"],
        "price": [1.5, 2.5, 3.5],
    }
)
prices = LabelledArray.from_dataframe(df, cols=["price"], dims=["region", "product"], dtype="float32")

for name in available_backends():
    moved = prices.to_backend(name)
    print(f"{name:6} {type(moved['price']).__name__:12} {moved.dtypes['price']}")
    assert moved.equals(prices), "the values are unchanged by the move"

# %%
# Labels stay on numpy
# --------------------
# Only the variables move; everything you select and label with stays where it
# can hold strings and dates.

on_torch = prices.to_backend("torch") if "torch" in available_backends() else prices
assert isinstance(on_torch.dims["region"], np.ndarray)
assert on_torch.select({"region": "north"}).shape == (1, 2)
print(on_torch.to_dataframe(index=False))

# %%
# What a backend cannot hold
# --------------------------
# Torch and JAX have no string or datetime dtype. Rather than coercing quietly,
# the move fails and names the variable.

text = LabelledArray.from_dataframe(
    pd.DataFrame({"a": [1, 2], "label": ["p", "q"]}), cols=["label"], dims=["a"]
)
try:
    text.to_backend("torch")
except BackendDtypeError as error:
    print(error)

# %%
# JAX narrows 64-bit dtypes
# -------------------------
# Unless ``jax_enable_x64`` is set, JAX allocates 32 bits. This is JAX's own
# behaviour; the dtype reported is always the one actually allocated.

if "jax" in available_backends():
    import jax

    on_jax = prices.astype("float64").to_backend("jax")
    print("float64 became", on_jax.dtypes["price"], "(x64:", jax.config.read("jax_enable_x64"), ")")

# %%
# Dtypes through a round trip
# ---------------------------
# Because the mask records absence, a column does not have to be widened to
# hold a null: integers and booleans keep their dtype through a sparse frame.

typed = pd.DataFrame(
    {
        "region": ["north", "north", "south"],
        "product": ["AAA", "BBB", "AAA"],
        "count": np.array([1, 2, 3], dtype="int32"),
        "ok": np.array([True, False, True]),
    }
)
kept = LabelledArray.from_dataframe(typed, dims=["region", "product"])
assert kept.dtypes["count"] == np.dtype("int32")
assert kept.dtypes["ok"] == np.dtype("bool")
pd.testing.assert_frame_equal(
    typed.sort_values(["region", "product"]).reset_index(drop=True),
    kept.to_dataframe(index=False),
)
print(kept.dtypes)

# %%
# Where information is lost
# -------------------------
# Extension dtypes have no numpy equivalent, so they come back as the plain
# dtype underneath. README.md has the full table.

categorical = pd.DataFrame(
    {"a": [1, 2], "grade": pd.Categorical(["high", "low"], categories=["low", "high"], ordered=True)}
)
graded = LabelledArray.from_dataframe(categorical, cols=["grade"], dims=["a"])
print("category ->", graded.dtypes["grade"], "->", graded.to_dataframe()["grade"].dtype)
assert graded.to_dataframe()["grade"].tolist() == ["high", "low"], "values survive, categories do not"
