"""DataFrame -> LabelledArray -> DataFrame round trips, for every engine."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import assert_frames_equal, build_frame, n_rows, sort_frame

from labelled_array import LabelledArray

DIMS = ["region", "product"]


def round_trip(engine, data, dims=DIMS, **kwargs):
    """Return the source frame (canonically sorted) and its round trip."""
    df = build_frame(engine, data)
    la = LabelledArray.from_dataframe(df, dims=dims, **kwargs)
    return sort_frame(engine, df, dims), la.to_dataframe(engine, index=False)


def test_sparse_frame_round_trips_exactly(engine, sales_data):
    assert_frames_equal(engine, *round_trip(engine, sales_data))


def test_dense_frame_round_trips_exactly(engine):
    data = {
        "region": ["north", "north", "south", "south"],
        "product": ["AAA", "BBB", "AAA", "BBB"],
        "units": [1, 2, 3, 4],
        "price": [1.5, 2.5, 3.5, 4.5],
    }
    assert_frames_equal(engine, *round_trip(engine, data))


def test_row_order_does_not_matter(engine, sales_data):
    shuffled = {name: list(reversed(values)) for name, values in sales_data.items()}
    df_sorted, out = round_trip(engine, shuffled)
    assert_frames_equal(engine, df_sorted, out)


def test_single_dimension_round_trips(engine):
    data = {"region": ["south", "north"], "units": [2, 1]}
    assert_frames_equal(engine, *round_trip(engine, data, dims=["region"]))


def test_three_dimensions_round_trip(engine):
    data = {
        "region": ["north", "north", "south"],
        "product": ["AAA", "BBB", "AAA"],
        "year": [2023, 2024, 2023],
        "units": [1, 2, 3],
    }
    assert_frames_equal(engine, *round_trip(engine, data, dims=["region", "product", "year"]))


def test_single_row_round_trips(engine):
    data = {"region": ["north"], "product": ["AAA"], "units": [1]}
    assert_frames_equal(engine, *round_trip(engine, data))


def test_empty_frame_round_trips(engine):
    data = {
        "region": np.array([], dtype=np.int64),
        "product": np.array([], dtype=np.float64),
        "units": np.array([], dtype=np.float64),
    }
    df_sorted, out = round_trip(engine, data)
    assert n_rows(engine, out) == 0
    assert_frames_equal(engine, df_sorted, out)


def test_round_trip_after_a_selection(engine, sales_data):
    df = build_frame(engine, sales_data)
    la = LabelledArray.from_dataframe(df, dims=DIMS)
    out = la.isel(region=[0]).to_dataframe(engine, index=False)
    assert n_rows(engine, out) == 3


def test_round_trip_through_every_array_backend(engine, backend):
    data = {
        "region": ["north", "north", "south"],
        "product": ["AAA", "BBB", "AAA"],
        "price": [1.5, 2.5, 3.5],
    }
    df = build_frame(engine, data)
    la = LabelledArray.from_dataframe(df, dims=DIMS, dtype="float32").to_backend(backend)
    out = la.to_dataframe(engine, index=False)
    assert_frames_equal(
        engine,
        sort_frame(engine, build_frame(engine, {**data, "price": np.float32(data["price"])}), DIMS),
        out,
    )


def test_round_trip_is_idempotent(engine, sales_data):
    df = build_frame(engine, sales_data)
    first = LabelledArray.from_dataframe(df, dims=DIMS)
    second = LabelledArray.from_dataframe(first.to_dataframe(engine, index=False), dims=DIMS)
    assert first.equals(second)


def test_round_trip_without_a_mask_reintroduces_filled_cells(engine, sales_data):
    """Without a mask the fill values become real rows, which is the lossy path."""
    df = build_frame(engine, sales_data)
    la = LabelledArray.from_dataframe(df, dims=DIMS, mask=False, dtype="float64")
    assert n_rows(engine, la.to_dataframe(engine, index=False)) == 6
    assert n_rows(engine, la.to_dataframe(engine, index=False, dropna=True)) == 4


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("int64", [1, 2, 3]),
        ("int32", [1, 2, 3]),
        ("float64", [1.5, 2.5, 3.5]),
        ("float32", [1.5, 2.5, 3.5]),
        ("bool", [True, False, True]),
    ],
)
def test_dtypes_survive_a_sparse_round_trip(engine, dtype, values):
    """The mask is what lets int and bool columns keep their dtype."""
    data = {
        "region": ["north", "north", "south"],
        "product": ["AAA", "BBB", "AAA"],
        "units": np.array(values, dtype=dtype),
    }
    df = build_frame(engine, data)
    la = LabelledArray.from_dataframe(df, dims=DIMS)
    assert la.dtypes["units"] == np.dtype(dtype)
    assert_frames_equal(engine, sort_frame(engine, df, DIMS), la.to_dataframe(engine, index=False))
