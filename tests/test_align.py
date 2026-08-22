"""Putting two arrays onto one label set, and the arithmetic that needs it."""

from __future__ import annotations

import numpy as np
import pytest

from labelled_array import DimensionError, LabelledArray
from labelled_array._align import join_labels


@pytest.fixture
def left():
    return LabelledArray({"w": np.arange(3.0)}, {"b": ["x", "y", "z"]})


@pytest.fixture
def right():
    return LabelledArray({"w": np.array([10.0, 20.0])}, {"b": ["y", "q"]})


# -- joins ----------------------------------------------------------------------


def test_inner_keeps_the_labels_both_have(left, right):
    a, b = left.align(right)
    assert a.dims["b"].tolist() == ["y"] == b.dims["b"].tolist()
    assert a["w"].tolist() == [1.0]
    assert b["w"].tolist() == [10.0]
    assert a.mask is None and b.mask is None


def test_inner_keeps_the_order_of_the_first_array():
    a = LabelledArray({"w": np.arange(3.0)}, {"b": ["z", "y", "x"]})
    b = LabelledArray({"w": np.arange(3.0)}, {"b": ["x", "y", "z"]})
    assert a.align(b)[0].dims["b"].tolist() == ["z", "y", "x"]


def test_outer_takes_the_union_sorted(left, right):
    a, b = left.align(right, join="outer")
    assert a.dims["b"].tolist() == ["q", "x", "y", "z"]
    assert a.mask.tolist() == [False, True, True, True]
    assert b.mask.tolist() == [True, False, True, False]
    assert np.isnan(a["w"][0])


def test_left_and_right_take_one_array_labels(left, right):
    a, b = left.align(right, join="left")
    assert a.dims["b"].tolist() == ["x", "y", "z"] == b.dims["b"].tolist()
    assert a.equals(left), "the left array is untouched by a left join"

    a, b = left.align(right, join="right")
    assert a.dims["b"].tolist() == ["y", "q"]
    assert b.equals(right)


def test_exact_passes_when_the_labels_already_match(left):
    a, b = left.align(left.copy(), join="exact")
    assert a.equals(b)


def test_exact_raises_when_they_do_not(left, right):
    with pytest.raises(DimensionError, match="join='exact'"):
        left.align(right, join="exact")


def test_an_unknown_join_raises(left, right):
    with pytest.raises(ValueError, match="join must be one of"):
        left.align(right, join="nope")


def test_fill_value_is_used_for_the_cells_gained(left, right):
    a, _ = left.align(right, join="outer", fill_value=0.0)
    assert a["w"].tolist() == [0.0, 0.0, 1.0, 2.0]


def test_object_labels_are_matched_through_their_python_values():
    """Labels from a DataFrame are object arrays, which numpy cannot sort cheaply."""
    import pandas as pd

    left = LabelledArray.from_dataframe(
        pd.DataFrame({"k": ["a", "b"], "v": [1.0, 2.0]}), dims=["k"]
    )
    right = LabelledArray.from_dataframe(
        pd.DataFrame({"k": ["b", "c"], "v": [3.0, 4.0]}), dims=["k"]
    )
    assert left.dims["k"].dtype == np.dtype("O")
    assert left.align(right)[0].dims["k"].tolist() == ["b"]
    assert left.align(right, join="outer")[0].dims["k"].tolist() == ["a", "b", "c"]


def test_join_labels_rejects_an_unknown_join():
    with pytest.raises(ValueError, match="join must be one of"):
        join_labels("b", [np.array([1]), np.array([2])], "nope")


# -- what alignment leaves alone ------------------------------------------------


def test_matching_labels_are_left_untouched(left):
    other = left.copy()
    a, b = left.align(other)
    assert a is left and b is other, "nothing to join means nothing to copy"


def test_dimensions_the_other_array_does_not_have_are_kept(left):
    other = LabelledArray({"w": np.arange(6.0).reshape(3, 2)}, {"b": ["x", "y", "z"], "t": [0, 1]})
    a, b = left.align(other)
    assert a.dims.names == ("b",)
    assert b.dims.names == ("b", "t")
    assert b.shape == (3, 2)


def test_alignment_does_not_transpose():
    a = LabelledArray({"w": np.zeros((2, 3))}, {"b": ["x", "y"], "t": [0, 1, 2]})
    b = LabelledArray({"w": np.zeros((3, 2))}, {"t": [0, 1, 2], "b": ["x", "y"]})
    assert a.align(b)[1].dims.names == ("t", "b")


def test_more_than_two_arrays(left, right):
    third = LabelledArray({"w": np.zeros(2)}, {"b": ["y", "z"]})
    a, b, c = left.align(right, third)
    assert a.dims["b"].tolist() == ["y"]
    assert b.dims["b"].tolist() == ["y"] == c.dims["b"].tolist()


def test_aligning_one_array_is_a_no_op(left):
    assert left.align()[0] is left


def test_attributes_and_subclass_survive(left, right):
    class Custom(LabelledArray):
        pass

    custom = Custom(dict(left.data), left.dims, attrs={"k": 1})
    aligned, _ = custom.align(right, join="outer")
    assert isinstance(aligned, Custom)
    assert aligned.attrs == {"k": 1}


# -- arithmetic between arrays that do not line up ------------------------------


def test_arithmetic_aligns_on_the_shared_labels(left, right):
    total = left + right
    assert total.dims["b"].tolist() == ["y"]
    assert total["w"].tolist() == [11.0]


def test_arithmetic_aligns_a_transposed_operand():
    a = LabelledArray({"w": np.arange(6.0).reshape(2, 3)}, {"b": ["x", "y"], "t": [0, 1, 2]})
    doubled = a + a.transpose("t", "b")
    assert doubled.dims.names == ("b", "t")
    np.testing.assert_array_equal(doubled["w"], a["w"] * 2)


def test_arithmetic_can_be_told_to_join_outward(left, right):
    class Outer(LabelledArray):
        _arithmetic_join = "outer"

    total = Outer(dict(left.data), left.dims) + right
    assert total.dims["b"].tolist() == ["q", "x", "y", "z"]
    assert total.mask.tolist() == [False, False, True, False]
    assert total["w"][2] == 11.0


def test_arithmetic_still_refuses_different_dimension_names(left):
    other = LabelledArray({"w": np.arange(3.0)}, {"other": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="different dimensions"):
        left + other


def test_arithmetic_with_a_scalar_does_not_align(left):
    assert (left + 1)["w"].tolist() == [1.0, 2.0, 3.0]


def test_alignment_marks_the_invented_cells_absent(left, right):
    class Outer(LabelledArray):
        _arithmetic_join = "outer"

    total = Outer(dict(left.data), left.dims) + right
    assert np.isnan(total["w"][0]), "a cell neither operand had is not a number"
    assert not total.mask[0]


def test_reflected_arithmetic_aligns_too(left, right):
    assert (2.0 * right).dims["b"].tolist() == ["y", "q"]
    assert (left - right)["w"].tolist() == [1.0 - 10.0]
