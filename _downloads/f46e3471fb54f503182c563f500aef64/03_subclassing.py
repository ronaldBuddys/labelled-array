"""
.. _subclassing_example:

``LabelledArray`` : subclassing for a domain
============================================

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
import polars as pl
from polars.testing import assert_frame_equal

from labelled_array import DimensionError, LabelledArray


class ClimateArray(LabelledArray):
    """A labelled array of station readings, which must be indexed by date."""

    #: Replaced dimensions become e.g. 'date_pos' rather than 'date_idx'.
    _idx_suffix = "_pos"

    def _validate(self) -> None:
        super()._validate()
        if not any(name.startswith("date") for name in self.dims.names):
            raise DimensionError("a ClimateArray needs a date dimension")

    @property
    def dates(self) -> np.ndarray:
        """The date labels, or the dates a natural index stands in for."""
        return self.coords["date"] if "date" in self.coords else self.dims["date"]

    def anomaly(self) -> "ClimateArray":
        """Departure of every variable from its own mean over the date axis."""
        axis = self.dims.axis(next(n for n in self.dims.names if n.startswith("date")))
        data = {
            name: values - np.nanmean(values, axis=axis, keepdims=True)
            for name, values in self.data.items()
        }
        return self._replace(data=data)


# %%
# It behaves like any other labelled array
# ----------------------------------------

days = np.array(["2024-07-01", "2024-07-02", "2024-07-03"], dtype="datetime64[D]")
readings = pl.DataFrame(
    {
        "station": ["Oslo"] * 3 + ["Lisbon"] * 3,
        "date": np.tile(days, 2),
        "temperature": [17.0, 19.0, 21.0, 27.0, 31.0, 29.0],
    }
)
july = ClimateArray.from_dataframe(
    readings, cols=["temperature"], dims=["station", "date"]
)

print(july)
assert isinstance(july, ClimateArray)
assert july.dates.dtype.kind == "M"

# %%
# Validation is enforced at construction
# --------------------------------------

try:
    ClimateArray({"temperature": np.zeros(2)}, {"station": ["Oslo", "Lisbon"]})
except DimensionError as error:
    print(error)

# %%
# Derived arrays keep the subclass
# --------------------------------
# Selection, arithmetic, transposition, a backend move and the subclass's own
# method all return a ``ClimateArray``.

for operation in (
    lambda c: c.select({"station": "Oslo"}),
    lambda c: c.isel(date=0),
    lambda c: c.transpose(),
    lambda c: c * 2,
    lambda c: c.astype("float32"),
    lambda c: c.copy(),
    lambda c: c.to_backend("numpy"),
    lambda c: c.anomaly(),
):
    assert type(operation(july)) is ClimateArray

# Labels are sorted, so Lisbon is the first row. Each station's anomalies are
# measured against its own July mean, and so sum to nothing.
print(july.anomaly()["temperature"])
assert july.anomaly()["temperature"].tolist() == [[-2.0, 2.0, 0.0], [-2.0, 0.0, 2.0]]

# %%
# Class attributes tune the behaviour
# -----------------------------------
# ``_idx_suffix`` changes what a natural index dimension is called, and
# ``to_dataframe`` follows it back.

positional = ClimateArray.from_dataframe(
    readings, cols=["temperature"], dims=["station", "date"], natural=["date"]
)
assert positional.dims.names == ("station", "date_pos")
assert_frame_equal(readings.sort(["station", "date"]), positional.to_dataframe())
print(positional.to_dataframe())

# %%
# Extending the package instead of the class
# ------------------------------------------
# Array backends and DataFrame engines are registries, so another array library
# or DataFrame library is added from outside without touching the core.

from labelled_array.adapters import available_adapters  # noqa: E402
from labelled_array.backends import available_backends  # noqa: E402

print("backends:", available_backends())
print("engines :", available_adapters())
