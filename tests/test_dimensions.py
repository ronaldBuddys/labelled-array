"""Tests for :class:`labelled_array.dimensions.Dimensions`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy import array  # noqa: F401  (needed to eval reprs)

from labelled_array import DimensionError, LabelNotFoundError
from labelled_array.dimensions import Dimensions

# -- construction ---------------------------------------------------------------


def test_from_mapping_preserves_order_and_shape(labels_2d):
    dims = Dimensions(labels_2d)
    assert dims.names == ("a", "b")
    assert dims.shape == (2, 3)
    assert dims.ndim == 2
    assert dims.size == 6


def test_from_sequence_of_pairs_matches_mapping(labels_2d):
    assert Dimensions(list(labels_2d.items())) == Dimensions(labels_2d)


def test_from_dimensions_copies_natural_flags():
    dims = Dimensions({"a": [0, 1]}, natural=["a"])
    assert Dimensions(dims) == dims
    assert Dimensions(dims).is_natural("a")


def test_natural_may_name_the_coordinate_it_stands_in_for():
    dims = Dimensions({"date_idx": [0, 1]}, natural={"date_idx": "date"})
    assert dims.is_natural("date_idx")
    assert dims.source("date_idx") == "date"
    assert Dimensions({"a": [0, 1]}, natural=["a"]).source("a") is None


def test_rename_remaps_the_natural_source():
    dims = Dimensions({"date_idx": [0, 1]}, natural={"date_idx": "date"})
    assert dims.rename({"date": "d"}).source("date_idx") == "d"
    assert dims.rename({"date_idx": "t"}).source("t") == "date"


def test_labels_are_read_only(labels_2d):
    dims = Dimensions(labels_2d)
    with pytest.raises(ValueError):
        dims["a"][0] = 99


def test_construction_copies_source_array():
    source = np.array([1, 2, 3])
    dims = Dimensions({"a": source})
    source[0] = 99
    assert dims["a"].tolist() == [1, 2, 3]


def test_scalar_labels_become_length_one():
    assert Dimensions({"a": 7}).shape == (1,)


def test_empty_dimension_is_allowed():
    dims = Dimensions({"a": []})
    assert dims.shape == (0,)
    assert dims.size == 0


def test_zero_dimensional_is_allowed():
    dims = Dimensions({})
    assert dims.shape == ()
    assert dims.ndim == 0
    assert dims.size == 1


@pytest.mark.parametrize(
    ("dims", "match"),
    [
        ({"a": [1, 1]}, "must be unique"),
        ({"a": ["x", "y", "x"]}, "must be unique"),
        ({"a": [[1, 2], [3, 4]]}, "one-dimensional"),
        ({"": [1]}, "must not be empty"),
    ],
)
def test_invalid_labels_raise(dims, match):
    with pytest.raises(DimensionError, match=match):
        Dimensions(dims)


def test_duplicate_names_raise():
    with pytest.raises(DimensionError, match="names must be unique"):
        Dimensions([("a", [1]), ("a", [2])])


def test_non_string_name_raises():
    with pytest.raises(DimensionError, match="must be str"):
        Dimensions({0: [1, 2]})


def test_unknown_natural_name_raises():
    with pytest.raises(DimensionError, match="not dimensions"):
        Dimensions({"a": [1]}, natural=["b"])


# -- alternative constructors ---------------------------------------------------


def test_from_shape_builds_natural_index():
    dims = Dimensions.from_shape((2, 3))
    assert dims.names == ("dim_0", "dim_1")
    assert dims["dim_1"].tolist() == [0, 1, 2]
    assert all(dims.is_natural(name) for name in dims.names)


def test_from_shape_with_names():
    assert Dimensions.from_shape((2, 3), ["a", "b"]).names == ("a", "b")


def test_from_shape_name_count_mismatch_raises():
    with pytest.raises(DimensionError, match="1 names for 2 dimensions"):
        Dimensions.from_shape((2, 3), ["a"])


# -- mapping interface ----------------------------------------------------------


def test_mapping_protocol(labels_2d):
    dims = Dimensions(labels_2d)
    assert len(dims) == 2
    assert list(dims) == ["a", "b"]
    assert "a" in dims and "zz" not in dims
    assert dims["b"].tolist() == ["x", "y", "z"]
    assert list(dims.keys()) == ["a", "b"]
    assert [v.tolist() for v in dims.values()] == [[1, 2], ["x", "y", "z"]]


def test_unknown_dimension_lookup_raises(labels_2d):
    dims = Dimensions(labels_2d)
    with pytest.raises(DimensionError, match="unknown dimension 'zz'"):
        dims["zz"]
    with pytest.raises(DimensionError, match="unknown dimension"):
        dims.axis("zz")
    with pytest.raises(DimensionError, match="unknown dimension"):
        dims.is_natural("zz")


def test_axis(labels_2d):
    dims = Dimensions(labels_2d)
    assert dims.axis("a") == 0
    assert dims.axis("b") == 1


# -- get_indexer ----------------------------------------------------------------


def test_get_indexer_preserves_requested_order(labels_2d):
    dims = Dimensions(labels_2d)
    assert dims.get_indexer("b", ["z", "x"]).tolist() == [2, 0]


def test_get_indexer_accepts_scalar(labels_2d):
    assert Dimensions(labels_2d).get_indexer("a", 2).tolist() == [1]


@pytest.mark.parametrize("labels", [["x"], ("x",), np.array(["x"]), "x"])
def test_get_indexer_accepts_label_containers(labels_2d, labels):
    assert Dimensions(labels_2d).get_indexer("b", labels).tolist() == [0]


def test_get_indexer_allows_repeats(labels_2d):
    assert Dimensions(labels_2d).get_indexer("a", [1, 1, 2]).tolist() == [0, 0, 1]


def test_get_indexer_empty_selection(labels_2d):
    result = Dimensions(labels_2d).get_indexer("a", [])
    assert result.tolist() == []
    assert result.dtype == np.intp


def test_get_indexer_missing_raises_naming_dim_and_labels(labels_2d):
    dims = Dimensions(labels_2d)
    with pytest.raises(LabelNotFoundError) as excinfo:
        dims.get_indexer("b", ["x", "nope", "also-nope"])
    assert excinfo.value.dim == "b"
    assert excinfo.value.missing == ["nope", "also-nope"]
    assert "'b'" in str(excinfo.value)


def test_get_indexer_missing_ignore_drops(labels_2d):
    dims = Dimensions(labels_2d)
    assert dims.get_indexer("b", ["z", "nope", "x"], missing="ignore").tolist() == [
        2,
        0,
    ]


def test_get_indexer_is_a_keyerror(labels_2d):
    with pytest.raises(KeyError):
        Dimensions(labels_2d).get_indexer("a", [99])


def test_get_indexer_rejects_unknown_missing_mode(labels_2d):
    with pytest.raises(ValueError, match="missing must be one of"):
        Dimensions(labels_2d).get_indexer("a", [1], missing="drop")


def test_get_indexer_rejects_two_dimensional_labels(labels_2d):
    with pytest.raises(DimensionError, match="one-dimensional"):
        Dimensions(labels_2d).get_indexer("a", [[1, 2]])


def test_get_indexer_does_not_truncate_long_strings():
    """A '<U1' label array must not match a longer query truncated to one char."""
    dims = Dimensions({"b": ["x", "y"]})
    with pytest.raises(LabelNotFoundError):
        dims.get_indexer("b", ["xyz"])


def test_get_indexer_matches_datetimes_across_units():
    dims = Dimensions(
        {"d": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")}
    )
    assert dims.get_indexer("d", [np.datetime64("2020-01-02")]).tolist() == [1]
    assert dims.get_indexer("d", ["2020-01-01"]).tolist() == [0]


def test_get_indexer_matches_int_labels_from_other_int_widths():
    dims = Dimensions({"a": np.array([10, 20], dtype=np.int64)})
    assert dims.get_indexer("a", np.array([20], dtype=np.int32)).tolist() == [1]


def test_get_indexer_matches_float_labels_from_ints():
    assert Dimensions({"a": [1.0, 2.0]}).get_indexer("a", [2]).tolist() == [1]


def test_get_indexer_caches_lookup(labels_2d):
    dims = Dimensions(labels_2d)
    dims.get_indexer("a", [1])
    assert "a" in dims._lookups
    assert dims.get_indexer("a", [2]).tolist() == [1]


# -- derived instances ----------------------------------------------------------


def test_subset_takes_positions(labels_2d):
    subset = Dimensions(labels_2d).subset({"b": [2, 0]})
    assert subset["b"].tolist() == ["z", "x"]
    assert subset["a"].tolist() == [1, 2]


def test_subset_keeps_natural_flag():
    dims = Dimensions({"a": [0, 1, 2]}, natural=["a"])
    assert dims.subset({"a": [0, 2]}).is_natural("a")


def test_subset_unknown_dimension_raises(labels_2d):
    with pytest.raises(DimensionError, match="unknown dimensions"):
        Dimensions(labels_2d).subset({"zz": [0]})


def test_rename(labels_2d):
    renamed = Dimensions(labels_2d).rename({"a": "A"})
    assert renamed.names == ("A", "b")
    assert renamed["A"].tolist() == [1, 2]


def test_rename_moves_natural_flag():
    dims = Dimensions({"a": [0, 1]}, natural=["a"]).rename({"a": "A"})
    assert dims.is_natural("A")


def test_rename_unknown_raises(labels_2d):
    with pytest.raises(DimensionError, match="unknown dimensions"):
        Dimensions(labels_2d).rename({"zz": "A"})


def test_rename_onto_existing_name_raises(labels_2d):
    with pytest.raises(DimensionError, match="names must be unique"):
        Dimensions(labels_2d).rename({"a": "b"})


def test_transpose(labels_2d):
    transposed = Dimensions(labels_2d).transpose(["b", "a"])
    assert transposed.names == ("b", "a")
    assert transposed.shape == (3, 2)


def test_transpose_requires_permutation(labels_2d):
    with pytest.raises(DimensionError, match="not a permutation"):
        Dimensions(labels_2d).transpose(["b"])


def test_drop(labels_2d):
    assert Dimensions(labels_2d).drop("a").names == ("b",)
    assert Dimensions(labels_2d).drop(["a", "b"]).names == ()


def test_drop_unknown_raises(labels_2d):
    with pytest.raises(DimensionError, match="unknown dimensions"):
        Dimensions(labels_2d).drop("zz")


def test_replace_labels_and_flag(labels_2d):
    replaced = Dimensions(labels_2d).replace("a", [0, 1, 2], natural=True)
    assert replaced.shape == (3, 3)
    assert replaced.names == ("a", "b")
    assert replaced.is_natural("a")
    assert not replaced.replace("a", [0, 1, 2], natural=False).is_natural("a")


def test_derived_instances_do_not_mutate_source(labels_2d):
    dims = Dimensions(labels_2d)
    dims.subset({"a": [0]})
    dims.rename({"a": "A"})
    dims.drop("a")
    assert dims.names == ("a", "b")
    assert dims.shape == (2, 3)


# -- comparison and display -----------------------------------------------------


def test_equality(labels_2d):
    assert Dimensions(labels_2d) == Dimensions(labels_2d)
    assert Dimensions(labels_2d) != Dimensions({"a": [1, 2], "b": ["x", "y"]})
    assert Dimensions(labels_2d) != Dimensions({"b": ["x", "y", "z"], "a": [1, 2]})
    assert Dimensions({"a": [1]}) != Dimensions({"a": [1]}, natural=["a"])
    assert Dimensions(labels_2d) != "not dimensions"


def test_equality_distinguishes_dtype():
    assert Dimensions({"a": np.array([1, 2], dtype="int32")}) != Dimensions(
        {"a": np.array([1, 2], dtype="int64")}
    )


def test_unhashable(labels_2d):
    with pytest.raises(TypeError):
        hash(Dimensions(labels_2d))


def test_repr_round_trips(labels_2d):
    dims = Dimensions(labels_2d, natural=["a"])
    assert eval(repr(dims)) == dims


@pytest.mark.parametrize(
    "labels",
    [
        np.arange(2000),
        np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        np.array([1.5, 2.5], dtype="float32"),
        np.array([True, False]),
        np.array(["a", "bb"], dtype=object),
    ],
)
def test_repr_round_trips_for_dtypes(labels):
    dims = Dimensions({"a": labels})
    assert eval(repr(dims)) == dims


def test_to_repr_not_full_is_summarised():
    summarised = Dimensions({"a": np.arange(2000)}).to_repr(full=False)
    assert "..." in summarised


def test_str_summarises(labels_2d):
    text = str(Dimensions(labels_2d, natural=["a"]))
    assert "a" in text and "[natural]" in text and "(3)" in text


def test_str_of_zero_dimensional():
    assert "scalar" in str(Dimensions({}))


def test_get_indexer_on_a_datetime_dimension_with_an_uncastable_label():
    """A label that is not a date at all is simply absent, not a cast failure."""
    dims = Dimensions({"d": np.array(["2020-01-01"], dtype="datetime64[ns]")})
    with pytest.raises(LabelNotFoundError):
        dims.get_indexer("d", ["not-a-date"])
    assert dims.get_indexer("d", ["not-a-date"], missing="ignore").tolist() == []


# -- alternative dimensions -----------------------------------------------------


def test_alternative_labels_may_repeat_where_a_dimension_may_not():
    dims = Dimensions({"a": [1, 2, 3]}, alt={"g": ("a", ["x", "x", "y"])})
    assert dims.alt["g"].tolist() == ["x", "x", "y"]
    with pytest.raises(DimensionError, match="must be unique"):
        Dimensions({"a": ["x", "x", "y"]})


def test_alt_indexer_orders_by_request_and_ascends_within_a_label():
    dims = Dimensions({"a": [1, 2, 3, 4]}, alt={"g": ("a", ["x", "y", "x", "y"])})
    assert dims.alt_indexer("g", ["y", "x"]).tolist() == [1, 3, 0, 2]


def test_alt_indexer_honours_the_missing_mode():
    dims = Dimensions({"a": [1, 2]}, alt={"g": ("a", ["x", "y"])})
    with pytest.raises(LabelNotFoundError, match="'g'"):
        dims.alt_indexer("g", "z")
    assert dims.alt_indexer("g", ["z", "y"], missing="ignore").tolist() == [1]
    with pytest.raises(ValueError, match="missing must be one of"):
        dims.alt_indexer("g", "x", missing="drop")


def test_alt_indexer_matches_a_datetime_label_across_units():
    dims = Dimensions(
        {"a": [1, 2]},
        alt={
            "g": ("a", np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[ns]"))
        },
    )
    assert dims.alt_indexer("g", np.datetime64("2020-01-01", "D")).tolist() == [0, 1]


def test_subset_carries_the_alternative_labels():
    dims = Dimensions({"a": [1, 2, 3]}, alt={"g": ("a", ["x", "y", "z"])})
    assert dims.subset({"a": np.array([2, 0])}).alt["g"].tolist() == ["z", "x"]


def test_a_subset_alternative_stays_read_only():
    dims = Dimensions({"a": [1, 2, 3]}, alt={"g": ("a", ["x", "y", "z"])})
    with pytest.raises(ValueError):
        dims.subset({"a": np.array([0])}).alt["g"][0] = "q"


def test_drop_removes_the_alternatives_of_the_dropped_dimension():
    dims = Dimensions(
        {"a": [1, 2], "b": [3, 4]},
        alt={"g": ("a", ["x", "y"]), "h": ("b", ["p", "q"])},
    )
    assert set(dims.drop("a").alt) == {"h"}


def test_transpose_leaves_the_alternatives_alone():
    dims = Dimensions({"a": [1, 2], "b": [3, 4]}, alt={"g": ("a", ["x", "y"])})
    flipped = dims.transpose(["b", "a"])
    assert flipped.alt["g"].tolist() == ["x", "y"]
    assert flipped.alt_ref("g") == "a"


def test_rename_remaps_both_the_alternative_and_its_dimension():
    dims = Dimensions({"a": [1, 2]}, alt={"g": ("a", ["x", "y"])})
    renamed = dims.rename({"a": "asset", "g": "group"})
    assert renamed.alt_ref("group") == "asset"


def test_renaming_an_alternative_onto_a_dimension_raises():
    dims = Dimensions({"a": [1, 2], "b": [3, 4]}, alt={"g": ("a", ["x", "y"])})
    with pytest.raises(DimensionError, match="unique after renaming"):
        dims.rename({"g": "b"})


def test_equality_compares_the_alternatives():
    plain = Dimensions({"a": [1, 2]})
    tagged = Dimensions({"a": [1, 2]}, alt={"g": ("a", ["x", "y"])})
    assert plain != tagged
    assert tagged != Dimensions({"a": [1, 2]}, alt={"g": ("a", ["x", "z"])})
    assert tagged == Dimensions({"a": [1, 2]}, alt={"g": ("a", ["x", "y"])})


def test_repr_round_trips_with_an_alternative():
    dims = Dimensions(
        {"a": [1, 2], "n": [0, 1]}, natural=["n"], alt={"g": ("a", ["x", "y"])}
    )
    assert eval(repr(dims)) == dims


def test_an_alternative_cannot_be_tied_to_a_natural_index():
    # from_dataframe already refused this; the constructor now agrees, so an
    # object that could not round-trip cannot be built either way.
    with pytest.raises(DimensionError, match="cannot be tied to a natural index"):
        Dimensions({"a": [0, 1]}, natural={"a": "date"}, alt={"g": ("a", ["x", "y"])})


def test_str_flags_an_alternative():
    text = str(Dimensions({"a": [1, 2]}, alt={"g": ("a", ["x", "y"])}))
    assert "[alt for 'a']" in text


def test_drop_alt_removes_only_the_named_alternatives():
    dims = Dimensions(
        {"a": [1, 2], "b": [3, 4]},
        alt={"g": ("a", ["x", "y"]), "h": ("b", ["p", "q"])},
    )
    assert set(dims.drop_alt("g").alt) == {"h"}
    assert set(dims.drop_alt(["g", "h"]).alt) == set()
    with pytest.raises(DimensionError, match="unknown alternative dimensions"):
        dims.drop_alt("nope")
