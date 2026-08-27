"""Carrying the last present value forward along a dimension."""

from __future__ import annotations

import numpy as np
import pytest

from labelled_array import DimensionError, LabelledArray

DATES = np.array(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
).astype("datetime64[D]")


def curve(mask, backend=None, tenors=(1.0, 2.0)):
    """A (date, tenor) rate curve, absent on the dates ``mask`` marks False."""
    mask = np.asarray(mask, dtype=bool)[:, None] & np.ones(len(tenors), dtype=bool)
    rates = np.arange(mask.size, dtype=np.float64).reshape(mask.shape)
    rates = np.where(mask, rates, np.nan)
    return LabelledArray(
        {"rate": rates},
        {"date": DATES[: mask.shape[0]], "tenor": list(tenors)},
        mask=mask,
        coords={"quarter": np.arange(mask.shape[0])[:, None]},
        backend=backend,
    )


def rates_of(array):
    """Return the 'rate' variable as numpy, whatever backend it is on."""
    return array.backend.to_numpy(array["rate"])


# -- what gets carried ----------------------------------------------------------


def test_a_single_interior_gap_is_filled(backend):
    filled = curve([True, False, True], backend=backend).ffill("date")
    np.testing.assert_array_equal(rates_of(filled)[1], rates_of(filled)[0])
    assert filled.mask.all()


def test_a_run_of_consecutive_gaps_all_take_the_same_value(backend):
    filled = curve([True, False, False, False], backend=backend).ffill("date")
    original = rates_of(curve([True, False, False, False], backend=backend))
    for row in range(4):
        np.testing.assert_array_equal(rates_of(filled)[row], original[0])
    assert filled.mask.all()


def test_limit_carries_no_further_than_it_says(backend):
    filled = curve([True, False, False], backend=backend).ffill("date", limit=1)
    assert filled.mask[:, 0].tolist() == [True, True, False]
    np.testing.assert_array_equal(rates_of(filled)[1], rates_of(filled)[0])
    assert np.isnan(rates_of(filled)[2]).all(), "beyond the limit the fill value stays"


def test_a_gap_before_the_first_present_position_stays_absent(backend):
    source = curve([False, False, True], backend=backend)
    filled = source.ffill("date")
    assert filled.mask[:, 0].tolist() == [False, False, True]
    np.testing.assert_array_equal(rates_of(filled)[:2], rates_of(source)[:2])


def test_a_fully_present_array_is_returned_unchanged(backend):
    dense = LabelledArray(
        {"rate": np.arange(6.0).reshape(3, 2)},
        {"date": DATES[:3], "tenor": [1.0, 2.0]},
        backend=backend,
    )
    assert dense.ffill("date").equals(dense)


def test_carrying_along_the_other_dimension_works_too(backend):
    array = LabelledArray(
        {"rate": np.array([[1.0, np.nan, 3.0]])},
        {"date": DATES[:1], "tenor": [1.0, 2.0, 5.0]},
        mask=np.array([[True, False, True]]),
        backend=backend,
    )
    filled = array.ffill("tenor")
    np.testing.assert_array_equal(rates_of(filled), np.array([[1.0, 1.0, 3.0]]))


# -- the three cell arrays move together ----------------------------------------


def test_variables_coordinates_and_the_mask_all_move_together(backend):
    source = curve([True, False, True, False], backend=backend)
    filled = source.ffill("date")

    assert filled.mask.all(), "every gap follows a present position"
    np.testing.assert_array_equal(rates_of(filled)[1], rates_of(source)[0])
    np.testing.assert_array_equal(rates_of(filled)[3], rates_of(source)[2])
    # A filled cell's coordinates name the cell its value came from.
    assert filled.coords["quarter"][:, 0].tolist() == [0, 0, 2, 2]
    assert filled.dims["date"].tolist() == source.dims["date"].tolist()


def test_the_dimensions_and_labels_are_untouched(backend):
    source = curve([True, False, True], backend=backend)
    filled = source.ffill("date")
    assert filled.shape == source.shape
    assert filled.dims.names == source.dims.names
    np.testing.assert_array_equal(filled.dims["date"], source.dims["date"])


def test_a_filled_array_emits_the_rows_the_fill_created(backend, engine):
    filled = curve([True, False, True], backend=backend).ffill("date")
    frame = filled.to_dataframe(engine=engine, index=False)
    assert len(frame) == filled.size, "filled cells are present, so they are emitted"


# -- what it refuses ------------------------------------------------------------


def test_an_unknown_dimension_raises_naming_it():
    with pytest.raises(DimensionError) as raised:
        curve([True, False, True]).ffill("maturity")
    assert "maturity" in str(raised.value)


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_raises(limit):
    with pytest.raises(ValueError, match="limit must be at least 1"):
        curve([True, False, True]).ffill("date", limit=limit)


# -- the staleness recipe -------------------------------------------------------


def test_the_staleness_recipe_reports_how_old_each_carried_curve_is(backend):
    source = curve([False, True, False, False, True], backend=backend)
    dates = source.dims["date"]

    present = source.mask.any(axis=-1)
    positions = np.arange(len(dates))
    last = np.maximum.accumulate(np.where(present, positions, -1))
    age = np.where(last >= 0, dates - dates[last], np.timedelta64("NaT", "D"))

    filled = source.ffill("date")._replace(
        coords={**source.coords, "fill_age": age[:, None]}
    )

    ages = filled.coords["fill_age"][:, 0]
    assert np.isnat(ages[0]), "nothing precedes the first date"
    assert ages[1:].astype("timedelta64[D]").astype(int).tolist() == [0, 1, 2, 0]
    assert filled.mask[:, 0].tolist() == [False, True, True, True, True]
