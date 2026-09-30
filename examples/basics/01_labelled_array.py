"""
.. _labelled_array_example:

``LabelledArray`` : building one from scratch
=============================================

A :class:`~labelled_array.LabelledArray` is one or more equally shaped arrays
plus an ordered mapping of dimension name to labels. This example builds one by
hand and shows what it can tell you about itself.
"""

# %%
# Building an array
# -----------------
# Variables are given as a mapping, and every one of them must have the shape
# implied by the dimensions.

import numpy as np

from labelled_array import (  # noqa: F401  (Dimensions is used by eval(repr(...)))
    Dimensions,
    LabelledArray,
)

rainfall = np.array([[12.0, 0.0, 3.5], [8.0, 1.5, 0.0]])
sunshine = np.array([[1.0, 9.5, 6.0], [3.0, 8.0, 9.0]])

weather = LabelledArray(
    data={"rainfall": rainfall, "sunshine": sunshine},
    dims={"city": ["Oslo", "Lisbon"], "month": ["Jan", "Feb", "Mar"]},
    attrs={"units": "mm and hours"},
)

print(weather)

# %%
# Inspecting it
# -------------
# Dimensions carry the labels; the variables carry the numbers.

print("shape     :", weather.shape)
print("dimensions:", weather.dims.names)
print("variables :", weather.variables)
print("dtypes    :", weather.dtypes)
print("months    :", weather.dims["month"].tolist())

assert weather.shape == (2, 3)
assert weather.size == 6
assert len(weather) == 2, "len counts variables, size counts cells"

# %%
# Getting at the data
# -------------------
# Indexing by name returns the underlying array; a list of names returns a
# smaller labelled array.

assert weather["rainfall"][0, 0] == 12.0
assert weather[["rainfall"]].variables == ("rainfall",)
assert np.asarray(weather[["rainfall"]]).shape == (2, 3)

# `values` is the shortcut for the single-variable case.
assert weather[["sunshine"]].values.shape == (2, 3)

# %%
# Default labels: a natural index
# -------------------------------
# When only the order matters, leave the dimensions out and each axis is
# labelled ``0 .. n-1``.

positional = LabelledArray({"w": np.arange(6).reshape(2, 3)})
print(positional.dims)
assert positional.dims.names == ("dim_0", "dim_1")
assert positional.dims.is_natural("dim_0")

# %%
# Arithmetic
# ----------
# Operators apply to every variable and keep the labels.

doubled = weather * 2
assert doubled["rainfall"][0, 0] == 24.0
assert doubled.dims == weather.dims

total = weather[["rainfall"]] + weather[["rainfall"]]
assert total["rainfall"][0, 0] == 24.0

# %%
# The repr recreates the object
# -----------------------------
# ``repr`` returns a constructor call, so a small array can be pasted into a
# test or an issue and evaluated back.

# noqa codes: E402 because the import sits where it is explained; F401 because
# it is referenced only inside the eval'd repr below.
from numpy import array  # noqa: E402, F401

restored = eval(repr(weather))
assert restored == weather
print(repr(weather[["rainfall"]]))
