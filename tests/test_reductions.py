"""Mask-aware reductions over named dimensions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labelled_array import DimensionError, LabelledArray


@pytest.fixture
def sales():
    """A sparse 2 x 3 array: 'south' only sells 'a'."""
    frame = pd.DataFrame(
        {
            "region": ["north", "north", "north", "south"],
            "product": ["a", "b", "c", "a"],
            "units": [1, 2, 3, 4],
            "price": [1.5, 2.5, 3.5, 4.5],
        }
    )
    return LabelledArray.from_dataframe(frame, dims=["region", "product"]), frame


@pytest.fixture
def dense():
    return LabelledArray(
        {"w": np.arange(6.0).reshape(2, 3), "k": np.arange(6).reshape(2, 3)},
        {"a": [1, 2], "b": ["x", "y", "z"]},
        attrs={"source": "fixture"},
    )


# -- which dimensions go ---------------------------------------------------------


def test_reducing_one_dimension_drops_it(dense):
    total = dense.sum("b")
    assert total.dims.names == ("a",)
    assert total.dims["a"].tolist() == [1, 2]
    assert total["w"].tolist() == [3.0, 12.0]


def test_reducing_several_dimensions(dense):
    assert dense.sum(["a", "b"]).shape == ()
    assert dense.sum(["a", "b"])["w"] == 15.0


def test_reducing_none_named_reduces_everything(dense):
    assert dense.sum().shape == ()
    assert dense.sum().dims.names == ()
    assert dense.mean()["w"] == 2.5


def test_reducing_no_dimensions_at_all_changes_nothing(dense):
    assert dense.reduce("sum", []).equals(dense)


def test_the_other_labels_are_untouched(dense):
    assert dense.max("a").dims["b"].tolist() == ["x", "y", "z"]


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_every_reduction_matches_numpy(dense, op):
    expected = getattr(np, op)(dense["w"], axis=1)
    np.testing.assert_allclose(getattr(dense, op)("b")["w"], expected)


def test_reduce_takes_the_op_by_name(dense):
    assert dense.reduce("max", "b").equals(dense.max("b"))


def test_an_unknown_op_raises(dense):
    with pytest.raises(ValueError, match="op must be one of"):
        dense.reduce("median", "b")


def test_an_unknown_dimension_raises(dense):
    with pytest.raises(DimensionError, match="unknown dimensions"):
        dense.sum("nope")


# -- only populated cells count --------------------------------------------------


def test_absent_cells_do_not_contribute(sales):
    array, frame = sales
    totals = array.sum("product")
    expected = frame.groupby("region")["units"].sum()
    assert (
        totals["units"].tolist() == expected.loc[array.dims["region"].tolist()].tolist()
    )


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_reductions_agree_with_pandas_on_a_sparse_array(sales, op):
    array, frame = sales
    reduced = getattr(array, op)("product")
    expected = getattr(frame.groupby("region")["price"], op)()
    np.testing.assert_allclose(
        reduced["price"], expected.loc[array.dims["region"].tolist()].to_numpy()
    )


def test_a_reduction_over_populated_cells_needs_no_mask(sales):
    array, _ = sales
    assert array.sum("product").mask is None, "every region has at least one row"


def test_a_cell_with_nothing_behind_it_is_absent():
    array = LabelledArray(
        {"w": np.arange(4.0).reshape(2, 2)},
        {"a": [1, 2], "b": ["x", "y"]},
        mask=np.array([[True, True], [False, False]]),
    )
    total = array.sum("b")
    assert total.mask.tolist() == [True, False]
    assert np.isnan(total["w"][1]), "an empty sum is absent, not zero"


def test_an_absent_integer_cell_keeps_the_dtype():
    array = LabelledArray(
        {"k": np.arange(4).reshape(2, 2)},
        {"a": [1, 2], "b": ["x", "y"]},
        mask=np.array([[True, True], [False, False]]),
    )
    total = array.sum("b")
    assert total.dtypes["k"] == np.dtype("int64")
    assert total["k"].tolist() == [1, 0]
    assert total.mask.tolist() == [True, False]


def test_a_null_that_is_masked_in_is_data():
    array = LabelledArray({"w": np.array([1.0, np.nan])}, {"b": ["x", "y"]})
    assert np.isnan(array.sum()["w"]), "the mask records absence; NaN is a value"


def test_the_mean_divides_by_the_populated_count(sales):
    array, _ = sales
    assert array.mean("product")["price"].tolist() == [(1.5 + 2.5 + 3.5) / 3, 4.5]


# -- dtypes ----------------------------------------------------------------------


def test_an_integer_sum_stays_integer(dense):
    assert dense.sum("b").dtypes["k"] == np.dtype("int64")


def test_an_integer_mean_becomes_float(dense):
    assert dense.mean("b").dtypes["k"] == np.dtype("float64")


def test_booleans_sum_to_integers():
    array = LabelledArray({"f": np.array([True, False, True])}, {"i": [0, 1, 2]})
    assert array.sum()["f"] == 2
    assert bool(array.min()["f"]) is False and bool(array.max()["f"]) is True


def test_dates_can_be_compared_but_not_summed():
    array = LabelledArray(
        {"t": np.array(["2024-01-02", "2024-01-01"], "datetime64[D]")}, {"i": [0, 1]}
    )
    assert array.min()["t"] == np.datetime64("2024-01-01")
    assert array.max()["t"] == np.datetime64("2024-01-02")
    with pytest.raises(TypeError, match="dates can be compared but not added"):
        array.sum()


def test_durations_can_be_averaged():
    array = LabelledArray({"d": np.array([1, 3], "timedelta64[D]")}, {"i": [0, 1]})
    assert array.mean()["d"] == np.timedelta64(2, "D")


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_strings_cannot_be_reduced(op):
    array = LabelledArray({"s": np.array(["a", "b"], dtype=object)}, {"i": [0, 1]})
    with pytest.raises(TypeError, match="dtype object"):
        array.reduce(op)


def test_complex_numbers_can_be_summed_but_not_ordered():
    array = LabelledArray({"z": np.array([1 + 2j, 3 + 4j])}, {"i": [0, 1]})
    assert array.sum()["z"] == 4 + 6j
    with pytest.raises(TypeError, match="dtype complex128"):
        array.max()


# -- what travels with the values ------------------------------------------------


def test_attributes_and_subclass_survive(dense):
    class Custom(LabelledArray):
        pass

    custom = Custom(dict(dense.data), dense.dims, attrs={"k": 1})
    assert isinstance(custom.sum("b"), Custom)
    assert custom.sum("b").attrs == {"k": 1}


def test_a_coordinate_constant_along_the_reduced_dimension_is_kept():
    array = LabelledArray(
        {"w": np.arange(6.0).reshape(2, 3)},
        {"a": [1, 2], "b": ["x", "y", "z"]},
        coords={"sector": np.array([["t", "t", "t"], ["f", "f", "f"]], dtype=object)},
    )
    reduced = array.sum("b")
    assert reduced.coords["sector"].shape == (2,)
    assert reduced.coords["sector"].tolist() == ["t", "f"]


def test_a_coordinate_that_varies_along_it_is_dropped():
    array = LabelledArray(
        {"w": np.arange(6.0).reshape(2, 3)},
        {"a": [1, 2], "b": ["x", "y", "z"]},
        coords={"day": np.arange(6).reshape(2, 3)},
    )
    assert list(array.sum("b").coords) == [], "there is no cell left for it to label"
    assert list(array.sum("a").coords) == [], "and it varies along 'a' too"


def test_a_natural_index_coordinate_survives_a_reduction_over_the_other_dimension():
    panel = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "date": np.array(["2024-01-01", "2024-01-02"] * 2, "datetime64[D]"),
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    array = LabelledArray.from_dataframe(
        panel, cols=["v"], dims=["g", "date"], natural=["date"]
    )
    reduced = array.sum("g")
    assert reduced.dims.names == ("date_idx",)
    assert reduced.coords["date"].tolist() == panel["date"].unique().tolist()
    assert reduced["v"].tolist() == [4.0, 6.0]


# -- backends --------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_reductions_agree_across_backends(op, backend):
    rng = np.random.default_rng(0)
    array = LabelledArray(
        {"w": rng.standard_normal((4, 5))},
        {"a": list("abcd"), "b": range(5)},
        mask=rng.random((4, 5)) > 0.3,
    )
    expected = getattr(array, op)("b")
    moved = getattr(array.to_backend(backend), op)("b")
    np.testing.assert_allclose(
        backend.to_numpy(moved["w"]), expected["w"], equal_nan=True, rtol=1e-6
    )
    assert moved.mask is None or moved.mask.tolist() == expected.mask.tolist()
