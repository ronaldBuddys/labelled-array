"""Trailing windows along a dimension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labelled_array import DimensionError, LabelledArray, Rolling


@pytest.fixture
def series():
    """Two rows of five positions, the shape a rolling window is meant for."""
    return LabelledArray(
        {"w": np.arange(10.0).reshape(2, 5)},
        {"a": ["p", "q"], "t": [0, 1, 2, 3, 4]},
        attrs={"source": "fixture"},
    )


# -- the windows -----------------------------------------------------------------


def test_a_window_covers_a_position_and_the_ones_before_it(series):
    rolled = series.rolling("t", 3).sum()
    assert rolled["w"][0].tolist()[2:] == [0 + 1 + 2, 1 + 2 + 3, 2 + 3 + 4]


def test_partial_windows_are_absent_by_default(series):
    rolled = series.rolling("t", 3).sum()
    assert np.isnan(rolled["w"][:, :2]).all()
    assert rolled.mask[:, :2].tolist() == [[False, False], [False, False]]
    assert rolled.mask[:, 2:].all()


def test_min_periods_lets_the_short_windows_through(series):
    rolled = series.rolling("t", 3, min_periods=1).sum()
    assert rolled.mask is None
    assert rolled["w"][0].tolist() == [0.0, 1.0, 3.0, 6.0, 9.0]


def test_a_window_of_one_keeps_every_value(series):
    assert series.rolling("t", 1).max().equals(series)


def test_a_window_as_long_as_the_dimension(series):
    rolled = series.rolling("t", 5).sum()
    assert rolled["w"][:, 4].tolist() == [10.0, 35.0]
    assert rolled.mask[:, :4].sum() == 0


def test_the_shape_labels_and_attributes_are_kept(series):
    rolled = series.rolling("t", 2).mean()
    assert rolled.shape == series.shape
    assert rolled.dims["t"].tolist() == [0, 1, 2, 3, 4]
    assert rolled.attrs == {"source": "fixture"}


def test_coordinates_are_kept_because_no_cell_moves():
    array = LabelledArray(
        {"w": np.arange(4.0).reshape(1, 4)},
        {"g": ["a"], "i": [0, 1, 2, 3]},
        coords={"date": np.arange(4).reshape(1, 4)},
    )
    assert array.rolling("i", 2).sum().coords["date"].tolist() == [[0, 1, 2, 3]]


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("window", [1, 2, 4])
def test_rolling_agrees_with_pandas(series, op, window):
    expected = getattr(
        pd.DataFrame(series["w"].T).rolling(window, min_periods=window), op
    )().to_numpy()
    rolled = getattr(series.rolling("t", window), op)()
    np.testing.assert_allclose(rolled["w"].T, expected, equal_nan=True)


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_rolling_agrees_with_pandas_on_partial_windows(series, op):
    expected = getattr(
        pd.DataFrame(series["w"].T).rolling(3, min_periods=1), op
    )().to_numpy()
    rolled = getattr(series.rolling("t", 3, min_periods=1), op)()
    np.testing.assert_allclose(rolled["w"].T, expected, equal_nan=True)


# -- absent cells ----------------------------------------------------------------


def test_absent_cells_do_not_contribute():
    array = LabelledArray(
        {"w": np.array([[1.0, 2.0, 3.0]])},
        {"a": ["p"], "t": [0, 1, 2]},
        mask=np.array([[True, False, True]]),
    )
    rolled = array.rolling("t", 2, min_periods=1).sum()
    assert rolled["w"][0].tolist() == [1.0, 1.0, 3.0]


def test_a_window_short_of_populated_cells_is_absent():
    array = LabelledArray(
        {"w": np.array([[1.0, 2.0, 3.0]])},
        {"a": ["p"], "t": [0, 1, 2]},
        mask=np.array([[True, False, True]]),
    )
    rolled = array.rolling("t", 2).sum()
    assert rolled.mask.tolist() == [[False, False, False]]
    assert np.isnan(rolled["w"]).all(), "no window holds two populated cells"


def test_an_integer_variable_keeps_its_dtype_where_a_window_is_short():
    array = LabelledArray({"k": np.arange(3)}, {"t": [0, 1, 2]})
    rolled = array.rolling("t", 2).sum()
    assert rolled.dtypes["k"] == np.dtype("int64")
    assert rolled["k"].tolist() == [0, 1, 3]
    assert rolled.mask.tolist() == [False, True, True]


def test_rolling_over_a_natural_index_dimension():
    panel = pd.DataFrame(
        {
            "g": ["a", "a", "a", "b"],
            "date": np.array(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"],
                "datetime64[D]",
            ),
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    array = LabelledArray.from_dataframe(
        panel, cols=["v"], dims=["g", "date"], natural=["date"]
    )
    rolled = array.rolling("date_idx", 2, min_periods=1).mean()
    assert rolled.dims.names == ("g", "date_idx")
    assert rolled["v"][0].tolist() == [1.0, 1.5, 2.5]
    assert rolled["v"][1][:2].tolist() == [4.0, 4.0], (
        "b's second window still holds one date"
    )
    assert rolled.mask[1].tolist() == [True, True, False]
    assert np.isnan(rolled["v"][1][2]), "the last window of b holds nothing at all"


# -- backends and subclasses -----------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_rolling_agrees_across_backends(series, op, backend):
    expected = getattr(series.rolling("t", 3, min_periods=1), op)()
    rolled = getattr(series.to_backend(backend).rolling("t", 3, min_periods=1), op)()
    np.testing.assert_allclose(backend.to_numpy(rolled["w"]), expected["w"], rtol=1e-6)


def test_the_subclass_is_kept(series):
    class Custom(LabelledArray):
        pass

    custom = Custom(dict(series.data), series.dims)
    assert isinstance(custom.rolling("t", 2).sum(), Custom)


def test_the_class_is_importable_from_the_package(series):
    assert isinstance(series.rolling("t", 2), Rolling)


def test_repr_names_the_window(series):
    assert repr(series.rolling("t", 3)) == (
        "Rolling(LabelledArray over 't', window=3, min_periods=3)"
    )


# -- errors ----------------------------------------------------------------------


def test_an_unknown_dimension_is_refused(series):
    with pytest.raises(DimensionError, match="unknown dimensions"):
        series.rolling("nope", 2)


@pytest.mark.parametrize("window", [0, -1, 6])
def test_an_impossible_window_is_refused(series, window):
    with pytest.raises(DimensionError, match="window must be between 1 and"):
        series.rolling("t", window)


@pytest.mark.parametrize("min_periods", [0, 4])
def test_min_periods_outside_the_window_is_refused(series, min_periods):
    with pytest.raises(ValueError, match="min_periods must be between 1 and 3"):
        series.rolling("t", 3, min_periods=min_periods)


def test_an_unknown_op_is_refused(series):
    with pytest.raises(ValueError, match="op must be one of"):
        series.rolling("t", 2).reduce("median")


def test_strings_cannot_be_rolled():
    array = LabelledArray({"s": np.array(["a", "b"], dtype=object)}, {"t": [0, 1]})
    with pytest.raises(TypeError, match="dtype object"):
        array.rolling("t", 2).sum()
