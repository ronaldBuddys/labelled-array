"""Dtype coverage: what survives a round trip, and exactly how the rest is lost.

Every case here either round-trips exactly or asserts the documented lossy
behaviour, so nothing degrades silently.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from helpers import assert_frames_equal, build_frame, sort_frame
from labelled_array import BackendDtypeError, DimensionError, LabelledArray

DIMS = ["a", "b"]
#: A sparse key: (1, 'y') has no row, so the fill and mask paths are exercised.
KEY = {"a": [1, 1, 2], "b": ["x", "y", "x"]}


def round_trip(engine, values, **kwargs):
    """Round trip a single value column, returning the sorted source and result."""
    df = build_frame(engine, {**KEY, "v": values})
    la = LabelledArray.from_dataframe(df, dims=DIMS, **kwargs)
    return df, la, la.to_dataframe(engine, index=False)


# -- dtypes that survive exactly ------------------------------------------------

NUMERIC_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_numeric_dtypes_round_trip_exactly(engine, dtype):
    """The presence mask is what lets integers keep their dtype despite gaps."""
    values = np.array([1, 2, 3], dtype=dtype)
    df, la, out = round_trip(engine, values)
    assert la.dtypes["v"] == np.dtype(dtype)
    assert_frames_equal(engine, sort_frame(engine, df, DIMS), out)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array([True, False, True]), id="bool"),
        pytest.param(np.array([1.5, np.nan, 3.5]), id="float_with_nan"),
        pytest.param(np.array([np.nan] * 3), id="all_null"),
        pytest.param(["p", "q", "r"], id="str"),
        pytest.param(["", "q", "r"], id="str_with_empty"),
    ],
)
def test_other_dtypes_round_trip_exactly(engine, values):
    df, _, out = round_trip(engine, values)
    assert_frames_equal(engine, sort_frame(engine, df, DIMS), out)


def test_pandas_complex_round_trips():
    """polars has no complex dtype, so this is a pandas-only case."""
    df, _, out = round_trip("pandas", np.array([1 + 2j, 2 + 3j, 3 + 4j]))
    assert_frames_equal("pandas", sort_frame("pandas", df, DIMS), out)


def test_float16_round_trips(engine):
    df, la, out = round_trip(engine, np.array([1.5, 2.5, 3.5], dtype="float16"))
    assert la.dtypes["v"] == np.dtype("float16")
    assert_frames_equal(engine, sort_frame(engine, df, DIMS), out)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_pandas_datetimes_keep_their_unit(unit):
    values = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).astype(
        f"datetime64[{unit}]"
    )
    df, la, out = round_trip("pandas", values)
    assert la.dtypes["v"] == np.dtype(f"datetime64[{unit}]")
    assert_frames_equal("pandas", sort_frame("pandas", df, DIMS), out)


def test_pandas_timedeltas_round_trip():
    df, la, out = round_trip("pandas", pd.to_timedelta([1, 2, 3], unit="D"))
    assert la.dtypes["v"].kind == "m"
    assert_frames_equal("pandas", sort_frame("pandas", df, DIMS), out)


def test_pandas_object_column_of_mixed_types_round_trips():
    df, _, out = round_trip("pandas", np.array([1, "q", 3.5], dtype=object))
    assert_frames_equal("pandas", sort_frame("pandas", df, DIMS), out)


def test_polars_dates_and_durations_round_trip():
    for values in (
        [datetime.date(2024, 1, i + 1) for i in range(3)],
        [datetime.datetime(2024, 1, i + 1) for i in range(3)],
        [datetime.timedelta(days=i + 1) for i in range(3)],
    ):
        df, _, out = round_trip("polars", values)
        assert_frames_equal("polars", sort_frame("polars", df, DIMS), out)


# -- documented lossy paths -----------------------------------------------------


def test_pandas_timezone_aware_datetimes_survive_as_objects():
    """Values and dtype are preserved, but the array holds Timestamp objects."""
    values = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).tz_localize(
        "UTC"
    )
    df, la, out = round_trip("pandas", values)
    assert la.dtypes["v"] == np.dtype("O"), "numpy has no tz-aware dtype"
    assert_frames_equal("pandas", sort_frame("pandas", df, DIMS), out)
    # Without torch this would pass on the missing-backend error rather than the
    # dtype error it is meant to prove, so skip rather than assert vacuously.
    pytest.importorskip("torch")
    with pytest.raises(BackendDtypeError):
        la.to_backend("torch")


def test_pandas_categoricals_lose_their_categories():
    values = pd.Categorical(["p", "q", "r"], categories=["r", "q", "p"], ordered=True)
    df, la, out = round_trip("pandas", values)
    assert la.dtypes["v"] == np.dtype("O")
    assert out["v"].dtype != df["v"].dtype
    assert out["v"].tolist() == ["p", "q", "r"], "the values themselves are intact"


def test_pandas_nullable_integers_become_plain_numpy_dtypes():
    df, la, out = round_trip("pandas", pd.array([1, 2, 3], dtype="Int64"))
    assert la.dtypes["v"] == np.dtype("int64")
    assert out["v"].dtype == np.dtype("int64") != df["v"].dtype
    assert out["v"].tolist() == [1, 2, 3]


def test_pandas_nullable_integers_holding_na_are_promoted_to_float():
    df, la, out = round_trip("pandas", pd.array([1, None, 3], dtype="Int64"))
    assert la.dtypes["v"] == np.dtype("float64"), "pandas widens on to_numpy, not us"
    assert np.isnan(out["v"].tolist()[1])


def test_pandas_nullable_booleans_become_objects():
    df, la, out = round_trip("pandas", pd.array([True, None, False], dtype="boolean"))
    assert la.dtypes["v"] == np.dtype("O")
    assert out["v"].dtype == np.dtype("O")


def test_pandas_string_extension_dtype_comes_back_as_the_default_string_dtype():
    df, la, out = round_trip("pandas", pd.array(["p", "q", "r"], dtype="string"))
    assert la.dtypes["v"] == np.dtype("O")
    assert out["v"].tolist() == ["p", "q", "r"]


def test_polars_categoricals_come_back_as_strings():
    for dtype in (pl.Categorical, pl.Enum(["p", "q", "r"])):
        df, la, out = round_trip("polars", pl.Series(["p", "q", "r"], dtype=dtype))
        assert la.dtypes["v"] == np.dtype("O")
        assert out.schema["v"] == pl.String
        assert out["v"].to_list() == ["p", "q", "r"], "the values themselves are intact"


def test_polars_null_columns_become_float():
    df, la, out = round_trip("polars", pl.Series([None] * 3, dtype=pl.Null))
    assert la.dtypes["v"].kind == "f"
    assert out.schema["v"] != pl.Null


def test_an_explicit_fixed_width_string_dtype_truncates(engine):
    _, la, _ = round_trip(engine, ["short", "muchlongerstring", "z"], dtype="<U3")
    assert la["v"].tolist() == [["sho", "muc"], ["z", ""]]


def test_column_dtypes_are_reported_when_a_fill_or_cast_fails(engine):
    with pytest.raises(ValueError, match=r"cannot build variable 'v' of dtype uint8"):
        round_trip(engine, np.array([1, 2, 3], dtype="uint8"), fill_value=-1)
    with pytest.raises(ValueError, match=r"cannot build variable 'v' of dtype int64"):
        round_trip(engine, ["p", "q", "r"], dtype="int64")


# -- dimension label dtypes -----------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param([3, 1, 2], id="int"),
        pytest.param([3.5, 1.5, 2.5], id="float"),
        pytest.param(["c", "a", "b"], id="str"),
        pytest.param([True, False, True], id="bool"),
        pytest.param([datetime.date(2024, 1, d) for d in (3, 1, 2)], id="date"),
    ],
)
def test_dimension_label_dtypes_round_trip(engine, labels):
    data = {"k": labels, "v": [1.0, 2.0, 3.0]}
    if labels == [True, False, True]:
        data = {"k": [True, False], "v": [1.0, 2.0]}
    df = build_frame(engine, data)
    la = LabelledArray.from_dataframe(df, dims=["k"])
    assert_frames_equal(
        engine, sort_frame(engine, df, ["k"]), la.to_dataframe(engine, index=False)
    )


def test_labels_are_sorted_whatever_the_row_order(engine):
    df = build_frame(engine, {"k": [3, 1, 2], "v": [3.0, 1.0, 2.0]})
    assert LabelledArray.from_dataframe(df, dims=["k"]).dims["k"].tolist() == [1, 2, 3]


def test_timezone_aware_labels_round_trip():
    dates = pd.to_datetime(["2024-01-02", "2024-01-01"]).tz_localize("UTC")
    df = pd.DataFrame({"k": dates, "v": [1.0, 2.0]})
    la = LabelledArray.from_dataframe(df, dims=["k"])
    assert la.dims["k"].dtype == np.dtype("O")
    assert_frames_equal(
        "pandas", sort_frame("pandas", df, ["k"]), la.to_dataframe(index=False)
    )


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(["c", None, "b"], id="none"),
        pytest.param([3.5, np.nan, 1.5], id="nan"),
        pytest.param(pd.to_datetime(["2024-01-01", None]), id="nat"),
    ],
)
def test_null_labels_are_refused(labels):
    df = pd.DataFrame({"k": labels, "v": [1.0] * len(labels)})
    with pytest.raises(DimensionError, match="must not be null"):
        LabelledArray.from_dataframe(df, dims=["k"])


def test_mixed_type_labels_that_cannot_be_sorted_keep_first_seen_order():
    df = pd.DataFrame(
        {"k": np.array(["b", 1, "a"], dtype=object), "v": [1.0, 2.0, 3.0]}
    )
    la = LabelledArray.from_dataframe(df, dims=["k"])
    assert la.dims["k"].tolist() == ["b", 1, "a"]
    assert la["v"].tolist() == [1.0, 2.0, 3.0]


def test_labels_of_a_dimension_may_repeat_across_dimensions(engine):
    df = build_frame(engine, {"a": ["x", "x"], "b": ["x", "y"], "v": [1.0, 2.0]})
    la = LabelledArray.from_dataframe(df, dims=DIMS)
    assert la.shape == (1, 2)


# -- backend dtype limits -------------------------------------------------------


@pytest.mark.parametrize("name", ["jax", "torch"])
@pytest.mark.parametrize(
    "values",
    [["p", "q", "r"], pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])],
)
def test_non_numeric_variables_cannot_move_to_an_array_backend(name, values):
    from labelled_array import BackendDtypeError

    _, la, _ = round_trip("pandas", values)
    with pytest.raises(BackendDtypeError, match="cannot hold dtype"):
        la.to_backend(name)


def test_jax_narrows_64_bit_variables():
    """A documented backend gotcha, not something the conversion does."""
    import jax

    _, la, _ = round_trip("pandas", np.array([1.5, 2.5, 3.5], dtype="float64"))
    moved = la.to_backend("jax")
    expected = "float64" if jax.config.read("jax_enable_x64") else "float32"
    assert moved.dtypes["v"] == np.dtype(expected)
