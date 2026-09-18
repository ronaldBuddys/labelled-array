"""
Subclassing for a domain
========================

``LabelledArray`` is meant to be inherited from. Every operation that returns a
new array builds it through ``type(self)``, so a subclass keeps its own type,
its extra API and its own validation without overriding anything.
"""

# %%
# A domain-specific array
# -----------------------
# Two hooks are all that is usually needed: ``_validate`` to state what the
# subclass requires, and ordinary properties and methods for its own API.

import numpy as np
import pandas as pd

from labelled_array import DimensionError, LabelledArray


class FinancialArray(LabelledArray):
    """A labelled array that must be indexed by date."""

    #: Replaced dimensions become e.g. 'date_pos' rather than 'date_idx'.
    _idx_suffix = "_pos"

    def _validate(self) -> None:
        super()._validate()
        if not any(name.startswith("date") for name in self.dims.names):
            raise DimensionError("a FinancialArray needs a date dimension")

    @property
    def dates(self) -> np.ndarray:
        """The date labels, or the dates a natural index stands in for."""
        return self.coords["date"] if "date" in self.coords else self.dims["date"]

    def returns(self) -> "FinancialArray":
        """Period-on-period change of every variable, along the date axis."""
        axis = self.dims.axis(next(n for n in self.dims.names if n.startswith("date")))
        data = {
            name: np.diff(values, axis=axis, prepend=np.nan) / values
            for name, values in self.data.items()
        }
        return self._replace(data=data)


# %%
# It behaves like any other labelled array
# ----------------------------------------

prices = pd.DataFrame(
    {
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "date": np.array(["2024-01-01", "2024-01-02"] * 2, dtype="datetime64[D]"),
        "close": [10.0, 11.0, 20.0, 24.0],
    }
)
book = FinancialArray.from_dataframe(prices, cols=["close"], dims=["ticker", "date"])

print(book)
assert isinstance(book, FinancialArray)
assert book.dates.dtype.kind == "M"

# %%
# Validation is enforced at construction
# --------------------------------------

try:
    FinancialArray({"close": np.zeros(2)}, {"ticker": ["AAA", "BBB"]})
except DimensionError as error:
    print(error)

# %%
# Derived arrays keep the subclass
# --------------------------------
# Selection, arithmetic, transposition, a backend move and a DataFrame
# conversion all return a ``FinancialArray``.

for operation in (
    lambda b: b.select({"ticker": "AAA"}),
    lambda b: b.isel(date=0),
    lambda b: b.transpose(),
    lambda b: b * 2,
    lambda b: b.astype("float32"),
    lambda b: b.copy(),
    lambda b: b.to_backend("numpy"),
    lambda b: b.returns(),
):
    assert type(operation(book)) is FinancialArray

print(book.returns()["close"])

# %%
# Class attributes tune the behaviour
# -----------------------------------
# ``_idx_suffix`` changes what a natural index dimension is called, and
# ``to_dataframe`` follows it back.

positional = FinancialArray.from_dataframe(
    prices, cols=["close"], dims=["ticker", "date"], natural=["date"]
)
assert positional.dims.names == ("ticker", "date_pos")
pd.testing.assert_frame_equal(
    prices.sort_values(["ticker", "date"]).reset_index(drop=True),
    positional.to_dataframe(index=False),
)
print(positional.to_dataframe(index=False))

# %%
# Extending the package instead of the class
# ------------------------------------------
# Array backends and DataFrame engines are registries, so another array library
# or DataFrame library is added from outside without touching the core.

from labelled_array.adapters import available_adapters  # noqa: E402
from labelled_array.backends import available_backends  # noqa: E402

print("backends:", available_backends())
print("engines :", available_adapters())
