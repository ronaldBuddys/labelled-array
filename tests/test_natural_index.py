"""Natural index dimensions: collapsing a near-unique dimension into positions."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import (
    assert_frames_equal,
    build_both,
    build_frame,
    column,
    columns,
    sort_frame,
)
from labelled_array import DimensionError, DuplicateIndexError, LabelledArray


@pytest.fixture
def panel():
    """Three groups of four rows, each with dates that appear in no other group.

    Labelling the dates directly gives a 3 x 4 x 12 array that is 8% populated;
    indexing them by position gives 3 x 4 x 1.
    """
    return {
        "A": np.repeat([1, 2, 3], 4).tolist(),
        "B": list("abcd") * 3,
        "date": np.arange(12).astype("datetime64[D]"),
        "w": np.arange(12, dtype=float).tolist(),
    }


DIMS = ["A", "B", "date"]


def test_natural_index_collapses_the_array(engine, panel):
    df = build_frame(engine, panel)
    labelled = LabelledArray.from_dataframe(df, cols=["w"], dims=DIMS)
    natural = LabelledArray.from_dataframe(df, cols=["w"], dims=DIMS, natural=["date"])

    assert labelled.shape == (3, 4, 12)
    assert natural.shape == (3, 4, 1)
    assert natural.size < labelled.size
    assert labelled.density == pytest.approx(12 / 144)
    assert natural.density == 1.0


def test_the_dimension_is_renamed_and_flagged(engine, panel):
    natural = LabelledArray.from_dataframe(
        build_frame(engine, panel), cols=["w"], dims=DIMS, natural=["date"]
    )
    assert natural.dims.names == ("A", "B", "date_idx")
    assert natural.dims.is_natural("date_idx")
    assert not natural.dims.is_natural("A")
    assert natural.dims["date_idx"].tolist() == [0]


def test_the_original_values_are_kept_as_a_coordinate(engine, panel):
    natural = LabelledArray.from_dataframe(
        build_frame(engine, panel), cols=["w"], dims=DIMS, natural=["date"]
    )
    assert list(natural.coords) == ["date"]
    assert natural.coords["date"].shape == natural.shape
    assert natural.coords["date"].dtype.kind == "M"


def test_round_trip_restores_the_original_column(engine, panel):
    df = build_frame(engine, panel)
    natural = LabelledArray.from_dataframe(df, cols=["w"], dims=DIMS, natural=["date"])
    out = natural.to_dataframe(engine, index=False)
    assert_frames_equal(engine, sort_frame(engine, df, DIMS), out)


def test_dimension_order_does_not_change_the_size(engine, panel):
    """The utils example asserted this; it holds because grouping ignores order."""
    df = build_frame(engine, panel)
    first = LabelledArray.from_dataframe(df, cols=["w"], dims=DIMS, natural=["date"])
    second = LabelledArray.from_dataframe(
        df, cols=["w"], dims=["date", "A", "B"], natural=["date"]
    )
    assert first.size == second.size
    assert second.shape == (1, 3, 4)


def test_ragged_groups_use_the_largest_and_mask_the_rest(engine):
    """Groups of different length still fit, with the short ones masked."""
    data = {
        "A": [1, 1, 1, 2],
        "date": np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"], "datetime64[D]"
        ),
        "w": [1.0, 2.0, 3.0, 4.0],
    }
    df = build_frame(engine, data)
    natural = LabelledArray.from_dataframe(
        df, cols=["w"], dims=["A", "date"], natural=["date"]
    )
    assert natural.shape == (2, 3)
    assert natural.mask.tolist() == [[True, True, True], [True, False, False]]
    assert_frames_equal(
        engine,
        sort_frame(engine, df, ["A", "date"]),
        natural.to_dataframe(engine, index=False),
    )


def test_natural_index_numbers_rows_within_each_group(engine):
    data = {
        "A": [1, 1, 2, 2],
        "date": np.array(
            ["2024-01-02", "2024-01-01", "2024-03-01", "2024-02-01"], "datetime64[D]"
        ),
        "w": [1.0, 2.0, 3.0, 4.0],
    }
    natural = LabelledArray.from_dataframe(
        build_frame(engine, data), cols=["w"], dims=["A", "date"], natural=["date"]
    )
    assert natural.shape == (2, 2)
    assert natural["w"].tolist() == [[2.0, 1.0], [4.0, 3.0]], (
        "ordered by date within group"
    )


def test_a_single_natural_dimension_becomes_a_global_rank(engine):
    data = {
        "date": np.array(["2024-01-02", "2024-01-01"], "datetime64[D]"),
        "w": [1.0, 2.0],
    }
    natural = LabelledArray.from_dataframe(
        build_frame(engine, data), cols=["w"], dims=["date"], natural=["date"]
    )
    assert natural.shape == (2,)
    assert natural["w"].tolist() == [2.0, 1.0]
    expected = np.array(["2024-01-01", "2024-01-02"], "datetime64[D]")
    np.testing.assert_array_equal(
        natural.coords["date"], expected.astype(natural.coords["date"].dtype)
    )


def test_several_natural_dimensions(engine):
    data = {
        "A": [1, 1, 2, 2],
        "b": ["p", "q", "r", "s"],
        "date": np.array(
            ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"], "datetime64[D]"
        ),
        "w": [1.0, 2.0, 3.0, 4.0],
    }
    df = build_frame(engine, data)
    natural = LabelledArray.from_dataframe(
        df, cols=["w"], dims=["A", "b", "date"], natural=["b", "date"]
    )
    assert natural.dims.names == ("A", "b_idx", "date_idx")
    assert sorted(natural.coords) == ["b", "date"]
    assert natural.size < 4 * 4 * 4
    assert_frames_equal(
        engine,
        sort_frame(engine, df, ["A", "b", "date"]),
        natural.to_dataframe(engine, index=False),
    )


def test_natural_accepts_a_single_name(engine, panel):
    la = LabelledArray.from_dataframe(
        build_frame(engine, panel), cols=["w"], dims=DIMS, natural="date"
    )
    assert la.dims.names == ("A", "B", "date_idx")


def test_unknown_natural_name_raises(engine, panel):
    with pytest.raises(DimensionError, match="natural names are not dimensions"):
        LabelledArray.from_dataframe(
            build_frame(engine, panel), cols=["w"], dims=DIMS, natural=["zz"]
        )


def test_expand_natural_false_writes_the_index_and_the_coordinate(engine, panel):
    natural = LabelledArray.from_dataframe(
        build_frame(engine, panel), cols=["w"], dims=DIMS, natural=["date"]
    )
    out = natural.to_dataframe(engine, index=False, expand_natural=False)
    assert columns(out) == ["A", "B", "date_idx", "date", "w"]
    assert column(out, "date_idx").tolist() == [0] * 12


def test_the_index_suffix_is_a_class_attribute(engine, panel):
    class Positional(LabelledArray):
        _idx_suffix = "_pos"

    la = Positional.from_dataframe(
        build_frame(engine, panel), cols=["w"], dims=DIMS, natural=["date"]
    )
    assert la.dims.names == ("A", "B", "date_pos")
    assert columns(la.to_dataframe(engine, index=False)) == ["A", "B", "date", "w"]


def test_selection_takes_the_coordinate_along(engine, panel):
    natural = LabelledArray.from_dataframe(
        build_frame(engine, panel), cols=["w"], dims=DIMS, natural=["date"]
    )
    selected = natural.isel(A=[0])
    assert selected.coords["date"].shape == selected.shape
    assert selected.coords["date"].tolist() == natural.coords["date"][:1].tolist()
    assert_frames_equal(
        engine,
        sort_frame(
            engine, build_frame(engine, {k: v[:4] for k, v in panel.items()}), DIMS
        ),
        selected.to_dataframe(engine, index=False),
    )


def test_string_labels_can_be_made_natural(engine):
    data = {"A": [1, 1, 2], "tag": ["z", "y", "x"], "w": [1.0, 2.0, 3.0]}
    df = build_frame(engine, data)
    natural = LabelledArray.from_dataframe(
        df, cols=["w"], dims=["A", "tag"], natural=["tag"]
    )
    assert natural.shape == (2, 2)
    assert_frames_equal(
        engine,
        sort_frame(engine, df, ["A", "tag"]),
        natural.to_dataframe(engine, index=False),
    )


def test_engines_agree_on_natural_indices(panel):
    built = build_both(panel, DIMS, cols=["w"], natural=["date"])
    assert built[0].equals(built[1])


def test_repr_of_a_natural_array_round_trips(engine, panel):
    from numpy import array  # noqa: F401

    from labelled_array import Dimensions  # noqa: F401

    natural = LabelledArray.from_dataframe(
        build_frame(engine, panel), cols=["w"], dims=DIMS, natural=["date"]
    )
    assert eval(repr(natural)) == natural


def test_renaming_keeps_the_natural_index_pointing_at_its_coordinate(engine, panel):
    """The dimension and its coordinate are linked by stored data, not by name."""
    df = build_frame(engine, panel)
    natural = LabelledArray.from_dataframe(df, cols=["w"], dims=DIMS, natural=["date"])

    renamed_dim = natural.rename({"date_idx": "position"})
    assert renamed_dim.dims.source("position") == "date"
    assert_frames_equal(
        engine,
        sort_frame(engine, df, DIMS),
        renamed_dim.to_dataframe(engine, index=False),
    )

    renamed_coord = natural.rename({"date": "day"})
    assert renamed_coord.dims.source("date_idx") == "day"
    assert columns(renamed_coord.to_dataframe(engine, index=False)) == [
        "A",
        "B",
        "day",
        "w",
    ]


def test_a_natural_dimension_without_a_coordinate_writes_out_its_index(engine, panel):
    """Dropping the coordinate leaves an ordinary positional dimension."""
    natural = LabelledArray.from_dataframe(
        build_frame(engine, panel), cols=["w"], dims=DIMS, natural=["date"]
    )
    positional = natural._replace(coords={})
    assert columns(positional.to_dataframe(engine, index=False)) == [
        "A",
        "B",
        "date_idx",
        "w",
    ]


def test_labels_repeating_within_a_group_are_refused(engine):
    # A long option chain: one row per strike, so the expiry repeats within a
    # trade date. Making *both* axes natural -- the surface shape -- groups the
    # expiry by trade_date alone, where it repeats once per strike. The index
    # counts rows, not distinct labels, so it would number every strike as its
    # own expiry: a plausible, wrong ladder of six rungs instead of two.
    chain = {
        "trade_date": ["2024-06-03"] * 6,
        "expiry": ["2024-06-21"] * 3 + ["2024-07-19"] * 3,
        "strike": [90.0, 100.0, 110.0] * 2,
        "vol": [0.2, 0.19, 0.21, 0.22, 0.2, 0.23],
    }
    df = build_frame(engine, chain)
    dims = ["trade_date", "expiry", "strike"]
    with pytest.raises(DuplicateIndexError, match="natural index 'expiry'") as info:
        LabelledArray.from_dataframe(
            df, cols=["vol"], dims=dims, natural=["expiry", "strike"]
        )
    message = str(info.value)
    assert "unique within ['trade_date']" in message
    assert "2024-06-21" in message

    # Declared the other way round, the strike is the one grouped by trade_date
    # alone, and it repeats too.
    with pytest.raises(DuplicateIndexError, match="natural index 'strike'"):
        LabelledArray.from_dataframe(
            df, cols=["vol"], dims=dims, natural=["strike", "expiry"]
        )

    # One natural axis is fine: the expiry is unique within (trade_date, strike)
    # and the strike within (trade_date, expiry), so either builds alone.
    one = LabelledArray.from_dataframe(df, cols=["vol"], dims=dims, natural=["expiry"])
    assert one.shape == (1, 2, 3)
    other = LabelledArray.from_dataframe(
        df, cols=["vol"], dims=dims, natural=["strike"]
    )
    assert other.shape == (1, 2, 3)


def test_a_forward_ladder_is_unique_per_group_and_builds(engine):
    # One row per (trade_date, expiry): the shape a forward curve has.
    ladder = {
        "trade_date": ["2024-06-03", "2024-06-03", "2024-06-04"],
        "expiry": ["2024-06-21", "2024-07-19", "2024-06-21"],
        "fwd": [100.0, 101.0, 100.5],
    }
    df = build_frame(engine, ladder)
    array = LabelledArray.from_dataframe(
        df, cols=["fwd"], dims=["trade_date", "expiry"], natural=["expiry"]
    )
    assert array.shape == (2, 2)
    assert array.mask.sum() == 3


def test_repeated_labels_with_no_grouping_dimension_are_refused(engine):
    df = build_frame(engine, {"d": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(DuplicateIndexError, match="natural index 'd'"):
        LabelledArray.from_dataframe(df, cols=["v"], dims=["d"], natural=["d"])
