"""Alternative dimensions: extra labels for a dimension's entries, not an axis."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from helpers import build_frame, column
from labelled_array import (
    DimensionError,
    Dimensions,
    LabelledArray,
    LabelNotFoundError,
)

ASSETS = ["AAA", "BBB", "CCC", "DDD"]
DATES = ["d1", "d2", "d3"]
SECTORS = ["tech", "tech", "fin", "tech"]


@pytest.fixture
def book():
    """A 4 x 3 book of prices whose assets carry a non-unique 'sector' label."""
    return LabelledArray(
        {"px": np.arange(12.0).reshape(4, 3)},
        dims={"asset": ASSETS, "date": DATES},
        alt_dims={"sector": {"ref_dim": "asset", "labels": SECTORS}},
    )


# -- declaring one --------------------------------------------------------------


def test_the_verbose_and_terse_spellings_agree(book):
    terse = LabelledArray(
        {"px": np.arange(12.0).reshape(4, 3)},
        dims={"asset": ASSETS, "date": DATES},
        alt_dims={"sector": ("asset", SECTORS)},
    )
    assert terse.equals(book)


def test_the_labels_need_not_be_unique(book):
    assert book.dims.alt["sector"].tolist() == SECTORS
    assert book.dims.alt_ref("sector") == "asset"
    assert book.dims.is_alt("sector")


def test_it_is_not_an_axis(book):
    assert book.shape == (4, 3)
    assert book.dims.names == ("asset", "date")
    assert "sector" not in book.dims


def test_the_labels_are_stored_once(book):
    assert book.dims.alt["sector"].shape == (4,)
    assert book.dims.alt["sector"].size == len(book.dims["asset"])


def test_stored_labels_are_read_only(book):
    with pytest.raises(ValueError):
        book.dims.alt["sector"][0] = "energy"


def test_the_wrong_number_of_labels_raises():
    with pytest.raises(DimensionError, match="has 2 labels, expected 4"):
        LabelledArray(
            {"px": np.arange(12.0).reshape(4, 3)},
            dims={"asset": ASSETS, "date": DATES},
            alt_dims={"sector": ("asset", ["tech", "fin"])},
        )


def test_an_unknown_reference_dimension_raises():
    with pytest.raises(DimensionError, match="unknown dimension 'ticker'"):
        LabelledArray(
            {"px": np.arange(12.0).reshape(4, 3)},
            dims={"asset": ASSETS, "date": DATES},
            alt_dims={"sector": ("ticker", SECTORS)},
        )


def test_null_labels_are_refused():
    with pytest.raises(DimensionError, match="must not be null"):
        LabelledArray(
            {"px": np.arange(12.0).reshape(4, 3)},
            dims={"asset": ASSETS, "date": DATES},
            alt_dims={"sector": ("asset", ["tech", None, "fin", "tech"])},
        )


def test_a_name_that_is_already_a_dimension_raises():
    with pytest.raises(DimensionError, match="'date' is already a dimension"):
        LabelledArray(
            {"px": np.arange(12.0).reshape(4, 3)},
            dims={"asset": ASSETS, "date": DATES},
            alt_dims={"date": ("asset", SECTORS)},
        )


def test_a_name_that_is_already_a_variable_raises():
    with pytest.raises(DimensionError, match="names must be unique"):
        LabelledArray(
            {"px": np.arange(12.0).reshape(4, 3)},
            dims={"asset": ASSETS, "date": DATES},
            alt_dims={"px": ("asset", SECTORS)},
        )


def test_a_name_that_is_already_a_coordinate_raises():
    with pytest.raises(DimensionError, match="names must be unique"):
        LabelledArray(
            {"px": np.arange(12.0).reshape(4, 3)},
            dims={"asset": ASSETS, "date": DATES},
            coords={"sector": np.zeros((4, 3))},
            alt_dims={"sector": ("asset", SECTORS)},
        )


def test_an_incomplete_specification_raises():
    with pytest.raises(DimensionError, match=r"missing \['labels'\]"):
        LabelledArray(
            {"px": np.arange(12.0).reshape(4, 3)},
            dims={"asset": ASSETS, "date": DATES},
            alt_dims={"sector": {"ref_dim": "asset"}},
        )


# -- selecting on one -----------------------------------------------------------


def test_one_label_keeps_every_entry_carrying_it(book):
    picked = book.select({"sector": "tech"})
    assert picked.shape == (3, 3)
    assert picked.dims["asset"].tolist() == ["AAA", "BBB", "DDD"]
    np.testing.assert_array_equal(picked["px"], book["px"][[0, 1, 3]])


def test_labels_come_back_in_the_order_requested(book):
    picked = book.select(sector=["fin", "tech"])
    assert picked.dims["asset"].tolist() == ["CCC", "AAA", "BBB", "DDD"]
    assert picked.dims.alt["sector"].tolist() == ["fin", "tech", "tech", "tech"]


def test_it_mixes_with_a_real_dimension_in_one_call(book):
    picked = book.select({"sector": "fin", "date": "d2"})
    assert picked.shape == (1, 1)
    assert picked["px"].item() == book["px"][2, 1]


def test_the_alternative_labels_are_subset_too(book):
    assert book.select(sector="tech").dims.alt["sector"].tolist() == ["tech"] * 3
    assert book.select(asset=["CCC", "AAA"]).dims.alt["sector"].tolist() == [
        "fin",
        "tech",
    ]


def test_drop_removes_the_dimension_and_its_alternative(book):
    dropped = book.select({"sector": "fin"}, drop=True)
    assert dropped.dims.names == ("date",)
    assert dropped.dims.alt == {}


def test_an_absent_label_raises(book):
    with pytest.raises(LabelNotFoundError, match="'sector'"):
        book.select(sector="energy")


def test_an_absent_label_can_be_ignored(book):
    picked = book.select({"sector": ["energy", "fin"]}, missing="ignore")
    assert picked.dims["asset"].tolist() == ["CCC"]


def test_selecting_twice_along_one_dimension_is_refused(book):
    with pytest.raises(DimensionError, match="both select along dimension 'asset'"):
        book.select({"sector": "tech", "asset": "AAA"})


def test_the_intersection_is_asked_for_as_two_calls(book):
    both = book.select(sector="tech").select(asset=["AAA", "DDD"])
    assert both.dims["asset"].tolist() == ["AAA", "DDD"]


def test_isel_does_not_see_it(book):
    with pytest.raises(DimensionError, match="unknown dimensions"):
        book.isel(sector=0)


# -- carried through ------------------------------------------------------------


def test_it_survives_isel(book):
    assert book.isel(asset=[3, 0]).dims.alt["sector"].tolist() == ["tech", "tech"]


def test_it_survives_transpose(book):
    flipped = book.transpose("date", "asset")
    assert flipped.dims.names == ("date", "asset")
    assert flipped.dims.alt["sector"].tolist() == SECTORS
    assert flipped.dims.alt_ref("sector") == "asset"


def test_it_survives_a_rename_of_its_own_name(book):
    renamed = book.rename(sector="industry")
    assert renamed.dims.alt["industry"].tolist() == SECTORS
    assert renamed.select(industry="fin").dims["asset"].tolist() == ["CCC"]


def test_it_follows_a_rename_of_its_dimension(book):
    renamed = book.rename(asset="ticker")
    assert renamed.dims.alt_ref("sector") == "ticker"
    assert renamed.select(sector="fin").dims["ticker"].tolist() == ["CCC"]


def test_it_survives_a_deep_copy(book):
    assert book.copy(deep=True).dims.alt["sector"].tolist() == SECTORS


def test_it_survives_a_pickle_round_trip(book):
    revived = pickle.loads(pickle.dumps(book))
    assert revived.dims.alt["sector"].tolist() == SECTORS
    assert revived.equals(book)


def test_it_is_part_of_equality(book):
    other = LabelledArray(
        {"px": np.arange(12.0).reshape(4, 3)},
        dims={"asset": ASSETS, "date": DATES},
        alt_dims={"sector": ("asset", ["fin"] * 4)},
    )
    assert not book.equals(other)
    assert not book.equals(book.rename(sector="industry"))


def test_it_survives_an_eval_able_repr(book):
    from numpy import array  # noqa: F401  -- the repr calls it

    assert eval(repr(book)).equals(book)


def test_it_shows_up_in_the_summary(book):
    assert "[alt for 'asset']" in str(book)


def test_it_survives_arithmetic(book):
    assert (book + book).dims.alt["sector"].tolist() == SECTORS


def test_it_survives_a_reindex_onto_a_subset(book):
    moved = book.reindex(asset=["DDD", "CCC"])
    assert moved.dims.alt["sector"].tolist() == ["tech", "fin"]


def test_it_survives_a_reindex_of_another_dimension(book):
    assert book.reindex(date=["d3", "d1"]).dims.alt["sector"].tolist() == SECTORS


# -- dropped ---------------------------------------------------------------------


def test_a_reindex_onto_new_labels_drops_it(book):
    # A label the array never had has no description, and a null one could never
    # be selected -- so the alternative dimension goes rather than being invented.
    assert book.reindex(asset=["AAA", "ZZZ"]).dims.alt == {}


def test_reducing_its_dimension_drops_it(book):
    assert book.sum("asset").dims.alt == {}
    assert book.sum("date").dims.alt["sector"].tolist() == SECTORS


def test_grouping_its_dimension_drops_it(book):
    grouped = book.groupby(np.array(SECTORS), dim="asset").mean()
    assert grouped.dims.alt == {}
    assert sorted(grouped.dims["asset"].tolist()) == ["fin", "tech"]


def test_concatenating_along_its_dimension_joins_it(book):
    other = LabelledArray(
        {"px": np.zeros((2, 3))},
        dims={"asset": ["EEE", "FFF"], "date": DATES},
        alt_dims={"sector": ("asset", ["energy", "fin"])},
    )
    joined = LabelledArray.concat([book, other], dim="asset")
    assert joined.dims["asset"].tolist() == [*ASSETS, "EEE", "FFF"]
    assert joined.dims.alt["sector"].tolist() == [*SECTORS, "energy", "fin"]
    assert joined.select(sector="fin").dims["asset"].tolist() == ["CCC", "FFF"]


def test_concatenating_along_another_dimension_keeps_it(book):
    later = book.reindex(date=["d4", "d5"])
    joined = LabelledArray.concat([book, later], dim="date")
    assert joined.dims.alt["sector"].tolist() == SECTORS
    assert joined.dims["date"].tolist() == [*DATES, "d4", "d5"]


def test_concatenating_arrays_that_disagree_about_it_raises(book):
    other = LabelledArray(
        {"px": np.zeros((4, 2))},
        dims={"asset": ASSETS, "date": ["d4", "d5"]},
        alt_dims={"sector": ("asset", ["fin"] * 4)},
    )
    with pytest.raises(DimensionError, match="disagree on alternative dimension"):
        LabelledArray.concat([book, other], dim="date")


def test_concatenating_an_array_without_it_raises(book):
    other = LabelledArray(
        {"px": np.zeros((4, 2))}, dims={"asset": ASSETS, "date": ["d4", "d5"]}
    )
    with pytest.raises(DimensionError, match="same alternative dimensions"):
        LabelledArray.concat([book, other], dim="date")


# -- grouping by one -------------------------------------------------------------


def test_grouping_by_it_needs_no_dimension_named(book):
    grouped = book.groupby("sector")
    assert grouped.dim == "asset"
    assert grouped.labels.tolist() == ["fin", "tech"]
    assert {
        label: positions.tolist() for label, positions in grouped.groups.items()
    } == {
        "fin": [2],
        "tech": [0, 1, 3],
    }


def test_grouping_by_it_reduces_within_each_group(book):
    summed = book.groupby("sector").sum()
    assert summed.dims["asset"].tolist() == ["fin", "tech"]
    np.testing.assert_array_equal(summed["px"][0], book["px"][2])
    np.testing.assert_array_equal(summed["px"][1], book["px"][[0, 1, 3]].sum(axis=0))


def test_grouping_by_it_matches_passing_the_values(book):
    assert (
        book.groupby("sector")
        .mean()
        .equals(book.groupby(np.array(SECTORS), dim="asset").mean())
    )


def test_the_result_no_longer_carries_it(book):
    # The dimension comes back labelled by group value, so the labels the
    # alternative dimension described are gone.
    assert book.groupby("sector").sum().dims.alt == {}


def test_naming_the_wrong_dimension_raises(book):
    with pytest.raises(DimensionError, match="describes 'asset', not 'date'"):
        book.groupby("sector", dim="date")


def test_naming_the_right_dimension_is_accepted(book):
    assert (
        book.groupby("sector", dim="asset").sum().equals(book.groupby("sector").sum())
    )


def test_an_unknown_name_lists_the_alternatives(book):
    with pytest.raises(DimensionError, match="unknown coordinate or alternative"):
        book.groupby("industry")


# -- the DataFrame round trip ----------------------------------------------------


def frame(engine, sectors=SECTORS):
    """A tidy frame of one row per asset and date, carrying a sector column."""
    return build_frame(
        engine,
        {
            "asset": [asset for asset in ASSETS for _ in DATES],
            "date": DATES * len(ASSETS),
            "sector": [sector for sector in sectors for _ in DATES],
            "px": list(np.arange(12.0)),
        },
    )


def test_from_dataframe_folds_a_column_in(engine):
    built = LabelledArray.from_dataframe(
        frame(engine), cols=["px"], dims=["asset", "date"], alt={"sector": "asset"}
    )
    assert built.shape == (4, 3)
    assert built.dims.alt["sector"].tolist() == SECTORS
    assert built.select(sector="fin").dims["asset"].tolist() == ["CCC"]


def test_a_third_dimension_would_have_been_mostly_empty(engine):
    spread = LabelledArray.from_dataframe(
        frame(engine), cols=["px"], dims=["asset", "date", "sector"]
    )
    folded = LabelledArray.from_dataframe(
        frame(engine), cols=["px"], dims=["asset", "date"], alt={"sector": "asset"}
    )
    assert spread.density < folded.density == 1.0


def test_to_dataframe_writes_it_back_out(engine, book):
    out = book.to_dataframe(engine=engine, index=False)
    assert sorted(set(column(out, "sector").tolist())) == ["fin", "tech"]
    pairs = set(
        zip(column(out, "asset").tolist(), column(out, "sector").tolist(), strict=True)
    )
    assert pairs == set(zip(ASSETS, SECTORS, strict=True))


def test_the_column_does_not_appear_when_there_is_no_alternative(engine):
    plain = LabelledArray({"px": np.zeros((4, 3))}, {"asset": ASSETS, "date": DATES})
    out = plain.to_dataframe(engine=engine, index=False)
    with pytest.raises((KeyError, Exception)):
        column(out, "sector")


def test_a_column_that_is_not_constant_per_entry_raises(engine):
    mixed = build_frame(
        engine,
        {
            "asset": ["AAA", "AAA"],
            "date": ["d1", "d2"],
            "sector": ["tech", "fin"],
            "px": [1.0, 2.0],
        },
    )
    with pytest.raises(DimensionError, match="not constant within dimension 'asset'"):
        LabelledArray.from_dataframe(
            mixed, cols=["px"], dims=["asset", "date"], alt={"sector": "asset"}
        )


def test_it_cannot_be_tied_to_a_natural_index(engine):
    with pytest.raises(DimensionError, match="cannot be tied to a natural index"):
        LabelledArray.from_dataframe(
            frame(engine),
            cols=["px"],
            dims=["asset", "date"],
            natural=["date"],
            alt={"sector": "date"},
        )


def test_an_unknown_column_raises(engine):
    with pytest.raises(KeyError, match="column not in the frame"):
        LabelledArray.from_dataframe(
            frame(engine),
            cols=["px"],
            dims=["asset", "date"],
            alt={"industry": "asset"},
        )


def test_an_unknown_reference_column_raises(engine):
    with pytest.raises(DimensionError, match="which is not one of the dimensions"):
        LabelledArray.from_dataframe(
            frame(engine),
            cols=["px"],
            dims=["asset", "date"],
            alt={"sector": "ticker"},
        )


def test_the_column_cannot_also_be_a_variable(engine):
    with pytest.raises(DimensionError, match="alternative dimensions and dimensions"):
        LabelledArray.from_dataframe(
            frame(engine),
            cols=["px", "sector"],
            dims=["asset", "date"],
            alt={"sector": "asset"},
        )


# -- the Dimensions object on its own --------------------------------------------


def test_alt_indexer_returns_every_matching_position():
    dims = Dimensions({"a": [1, 2, 3, 4]}, alt={"g": ("a", ["x", "y", "x", "y"])})
    np.testing.assert_array_equal(dims.alt_indexer("g", "x"), [0, 2])
    np.testing.assert_array_equal(dims.alt_indexer("g", ["y", "x"]), [1, 3, 0, 2])


def test_alt_indexer_rejects_an_unknown_alternative():
    dims = Dimensions({"a": [1, 2]}, alt={"g": ("a", ["x", "x"])})
    with pytest.raises(DimensionError, match="unknown alternative dimensions"):
        dims.alt_indexer("h", "x")


def test_alt_on_reports_what_is_tied_to_a_dimension():
    dims = Dimensions(
        {"a": [1, 2], "b": [3, 4]},
        alt={"g": ("a", ["x", "y"]), "h": ("b", ["p", "q"])},
    )
    assert set(dims.alt_on("a")) == {"g"}
    assert set(dims.alt_on("b")) == {"h"}


def test_replace_drops_them_unless_they_are_given():
    dims = Dimensions({"a": [1, 2]}, alt={"g": ("a", ["x", "y"])})
    assert dims.replace("a", [3, 4]).alt == {}
    assert dims.replace("a", [3, 4], alt={"g": ["p", "q"]}).alt["g"].tolist() == [
        "p",
        "q",
    ]


def test_a_non_string_name_raises():
    with pytest.raises(DimensionError, match="must be non-empty str"):
        Dimensions({"a": [1, 2]}, alt={3: ("a", ["x", "y"])})
    with pytest.raises(DimensionError, match="must be non-empty str"):
        Dimensions({"a": [1, 2]}, alt={"": ("a", ["x", "y"])})


def test_an_operand_that_lost_it_to_alignment_drops_it_for_everyone():
    """An outer join gives one array labels it has no description for."""
    left = LabelledArray(
        {"w": np.zeros((2, 2))},
        dims={"a": [1, 2], "d": ["p", "q"]},
        alt_dims={"g": ("a", ["x", "y"])},
    )
    right = LabelledArray(
        {"w": np.zeros((3, 2))},
        dims={"a": [1, 2, 3], "d": ["r", "s"]},
        alt_dims={"g": ("a", ["x", "y", "z"])},
    )
    # The outer join widens 'a' to [1, 2, 3]; 'left' has nothing to say about 3,
    # so its alternative dimension goes -- whichever operand comes first, the
    # join cannot keep one the other no longer has.
    for operands in ([left, right], [right, left]):
        joined = LabelledArray.concat(operands, dim="d", join="outer")
        assert joined.dims["a"].tolist() == [1, 2, 3]
        assert joined.dims.alt == {}


def test_layering_one_onto_existing_dimensions():
    dims = Dimensions({"a": [1, 2]})
    assert Dimensions(dims, alt={"g": ("a", ["x", "y"])}).alt["g"].tolist() == [
        "x",
        "y",
    ]
