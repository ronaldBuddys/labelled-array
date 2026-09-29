"""
.. _ffill_example:

``reindex``, ``ffill`` : forward filling, and knowing how stale the fill is
===========================================================================

Two calendars rarely agree. A discount curve is not published on every date an
option chain has a snapshot for, so putting the curve on the chain's dates
leaves gaps. The right answer is to carry the most recent curve forward -- and,
just as importantly, to know how old the one you used is, because a three-day
stale discount curve is a different fact about your forward than a same-day one.

``reindex`` does the first half. ``ffill`` does the carry. The age needs no new
concept at all: it is a coordinate.
"""

# %%
# A curve with a calendar of its own
# ----------------------------------
# Rates published on three dates, at two tenors.

import numpy as np

from labelled_array import LabelledArray

published = np.array(["2024-01-02", "2024-01-05", "2024-01-09"], "datetime64[D]")
tenors = [1.0, 5.0]

curve = LabelledArray(
    {"rate": np.array([[0.051, 0.043], [0.052, 0.044], [0.050, 0.042]])},
    {"date": published, "tenor": tenors},
)
print(curve)

# %%
# The dates we actually need it on
# --------------------------------
# ``reindex`` puts the curve on the option snapshot dates. The dates it was
# never published on are filled and marked absent, which is the gap to carry
# across.

snapshots = np.arange("2024-01-02", "2024-01-10", dtype="datetime64[D]")
on_snapshots = curve.reindex(date=snapshots)

print(on_snapshots.dims["date"])
print(on_snapshots.mask[:, 0])
assert on_snapshots.density < 1.0

# %%
# Recording the age before the fill
# ---------------------------------
# Take the mask *before* carrying anything: once the fill has happened, every
# cell is present and the gaps are gone. The running maximum of "my position if
# I am present, else -1" gives each date the position of the last published one
# at or before it.

present = on_snapshots.mask.any(axis=-1)
positions = np.arange(len(snapshots))
last = np.maximum.accumulate(np.where(present, positions, -1))
age = np.where(last >= 0, snapshots - snapshots[last], np.timedelta64("NaT", "D"))

print(dict(zip(snapshots.tolist(), age.tolist(), strict=True)))

# %%
# Carrying the curve forward
# --------------------------
# ``ffill`` walks the dimension in label order and gives each absent cell the
# last present value before it. Filled cells become present -- were they left
# absent, ``to_dataframe`` would drop the very rows the fill was performed to
# create.

filled = on_snapshots.ffill("date")

print(filled["rate"][:, 0])
assert filled.mask.all()
assert filled["rate"][2, 0] == filled["rate"][1, 0], "2024-01-04 carries 01-03's rate"

# %%
# The age is a coordinate, not a dimension
# ----------------------------------------
# It varies along ``date`` alone, so it is given at ``(n, 1)`` and stored once
# rather than repeated across every tenor. That is what makes tracking staleness
# free rather than an extra axis on the array.

filled = filled._replace(coords={**filled.coords, "fill_age": age[:, None]})

assert filled.coords["fill_age"].shape == filled.shape
assert filled.coords["fill_age"].base.nbytes == age.nbytes
print(filled.to_dataframe(index=False).head(8))

# %%
# Refusing to carry too far
# -------------------------
# ``limit`` caps how many positions a value travels. Beyond it, cells stay
# absent at their fill value -- which is how you say "a curve more than a day
# old is not a curve I will price against".

fresh = on_snapshots.ffill("date", limit=1)

print(fresh.mask[:, 0])
assert fresh.mask[:, 0].tolist() == [True, True, False, True, True, False, False, True]
assert fresh.density < 1.0

# %%
# A gap before the first observation stays a gap
# ----------------------------------------------
# There is nothing to carry, so those cells keep their fill value and stay
# absent. A forward fill invents nothing.

early = curve.reindex(date=np.arange("2023-12-31", "2024-01-04", dtype="datetime64[D]"))
carried = early.ffill("date")

print(carried.mask[:, 0])
assert carried.mask[:, 0].tolist() == [False, False, True, True]
