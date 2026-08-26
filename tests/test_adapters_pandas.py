"""pandas-specific adapter behaviour: the index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labelled_array import DimensionError, LabelledArray
from labelled_array.adapters import get_adapter


@pytest.fixture
def adapter():
    return get_adapter("pandas")


@pytest.fixture
def df(sales_data):
    return pd.DataFrame(sales_data)


def test_range_index_has_no_names(adapter, df):
    assert adapter.index_names(df) == []


def test_named_index_is_reported(adapter, df):
    assert adapter.index_names(df.set_index("region")) == ["region"]


def test_multiindex_names_are_reported(adapter, df):
    assert adapter.index_names(df.set_index(["region", "product"])) == [
        "region",
        "product",
    ]


def test_unnamed_multiindex_levels_are_skipped(adapter, df):
    indexed = df.set_index(["region", "product"])
    indexed.index.names = ["region", None]
    assert adapter.index_names(indexed) == ["region"]


def test_reset_index_moves_named_levels_into_columns(adapter, df):
    indexed = df.set_index(["region", "product"])
    assert adapter.column_names(adapter.reset_index(indexed)) == list(df.columns)


def test_reset_index_leaves_a_range_index_alone(adapter, df):
    assert adapter.reset_index(df) is df


def test_dims_default_to_the_index(df):
    from_index = LabelledArray.from_dataframe(df.set_index(["region", "product"]))
    from_columns = LabelledArray.from_dataframe(df, dims=["region", "product"])
    assert from_index.equals(from_columns)


def test_a_frame_with_no_named_index_needs_explicit_dims(df):
    with pytest.raises(DimensionError, match="no named index"):
        LabelledArray.from_dataframe(df)


def test_to_dataframe_builds_a_multiindex(df):
    out = LabelledArray.from_dataframe(df, dims=["region", "product"]).to_dataframe()
    assert isinstance(out.index, pd.MultiIndex)
    assert list(out.index.names) == ["region", "product"]
    assert list(out.columns) == ["units", "price"]


def test_to_dataframe_of_one_dimension_builds_a_plain_index():
    df = pd.DataFrame({"a": [1, 2], "w": [1.5, 2.5]})
    out = LabelledArray.from_dataframe(df, dims=["a"]).to_dataframe()
    assert out.index.name == "a"
    assert out.index.tolist() == [1, 2]


def test_to_dataframe_index_false_keeps_dimensions_as_columns(df):
    out = LabelledArray.from_dataframe(df, dims=["region", "product"]).to_dataframe(
        index=False
    )
    assert list(out.columns) == ["region", "product", "units", "price"]
    assert out.index.tolist() == [0, 1, 2, 3]


def test_index_round_trip(df):
    indexed = df.set_index(["region", "product"]).sort_index()
    out = LabelledArray.from_dataframe(indexed).to_dataframe()
    pd.testing.assert_frame_equal(indexed, out)


def test_default_engine_is_pandas(sales_data):
    import polars as pl

    la = LabelledArray.from_dataframe(
        pl.DataFrame(sales_data), dims=["region", "product"]
    )
    assert isinstance(la.to_dataframe(), pd.DataFrame), "the engine is not remembered"


def test_datetime_columns_keep_their_unit(adapter):
    """pandas 3 defaults to microsecond resolution, and the unit is preserved."""
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02", "2024-01-01"]), "w": [1.0, 2.0]}
    )
    values = adapter.column_values(df, "date")
    assert values.dtype == df["date"].dtype == np.dtype("datetime64[us]")
    out = LabelledArray.from_dataframe(df, dims=["date"]).to_dataframe(index=False)
    pd.testing.assert_frame_equal(df.sort_values("date").reset_index(drop=True), out)
