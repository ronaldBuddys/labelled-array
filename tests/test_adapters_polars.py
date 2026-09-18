"""polars-specific adapter behaviour: there is no index."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from labelled_array import DimensionError, LabelledArray
from labelled_array.adapters import get_adapter


@pytest.fixture
def adapter():
    return get_adapter("polars")


@pytest.fixture
def df(sales_data):
    return pl.DataFrame(sales_data)


def test_polars_frames_never_have_index_names(adapter, df):
    assert adapter.index_names(df) == []


def test_reset_index_is_a_no_op(adapter, df):
    assert adapter.reset_index(df) is df


def test_dims_must_be_given(df):
    with pytest.raises(DimensionError, match="no named index"):
        LabelledArray.from_dataframe(df)


def test_to_dataframe_ignores_index_and_keeps_dimension_columns(df):
    la = LabelledArray.from_dataframe(df, dims=["region", "product"])
    with_index = la.to_dataframe("polars", index=True)
    without_index = la.to_dataframe("polars", index=False)
    assert_frame_equal(with_index, without_index)
    assert with_index.columns == ["region", "product", "units", "price"]


def test_string_columns_arrive_as_object_arrays(adapter, df):
    """polars strings become object arrays, which is what the labels then hold."""
    values = adapter.column_values(df, "region")
    assert values.dtype == np.dtype("O")
    assert values.tolist() == ["north", "north", "south", "north"]


def test_string_labels_come_back_as_polars_strings(df):
    la = LabelledArray.from_dataframe(df, dims=["region", "product"])
    out = la.to_dataframe("polars")
    assert out.schema["region"] == pl.String


def test_round_trip(df):
    la = LabelledArray.from_dataframe(df, dims=["region", "product"])
    assert_frame_equal(df.sort(["region", "product"]), la.to_dataframe("polars"))


def test_date_columns_round_trip(adapter):
    import datetime

    df = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 2), datetime.date(2024, 1, 1)],
            "w": [1.0, 2.0],
        }
    )
    assert adapter.column_values(df, "date").dtype == np.dtype("datetime64[D]")
    out = LabelledArray.from_dataframe(df, dims=["date"]).to_dataframe("polars")
    assert out.schema["date"] == pl.Date
    assert_frame_equal(df.sort("date"), out)
