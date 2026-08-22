"""Moving an array onto a different label set, and filling what is missing."""

from __future__ import annotations

import numpy as np
import pytest

from labelled_array import DimensionError, LabelledArray


@pytest.fixture
def prices():
    """A 2 x 3 array of prices with one unpopulated cell."""
    return LabelledArray(
        {"price": np.arange(6.0).reshape(2, 3), "units": np.arange(6).reshape(2, 3)},
        {"region": ["north", "south"], "product": ["a", "b", "c"]},
        mask=np.array([[True, True, True], [True, True, False]]),
        attrs={"source": "fixture"},
    )


# -- labels the array already has -----------------------------------------------


def test_values_follow_their_labels(prices):
    moved = prices.reindex(product=["c", "a"])
    assert moved.dims["product"].tolist() == ["c", "a"]
    np.testing.assert_array_equal(moved["price"], prices["price"][:, [2, 0]])


def test_reindexing_can_drop_labels(prices):
    kept = prices.reindex(product=["b"])
    assert kept.shape == (2, 1)
    np.testing.assert_array_equal(kept["price"], prices["price"][:, [1]])


def test_reindexing_to_the_same_labels_changes_nothing(prices):
    assert prices.reindex(product=prices.dims["product"]).equals(prices)


def test_reindexing_nothing_returns_an_equal_array(prices):
    assert prices.reindex().equals(prices)


def test_several_dimensions_at_once(prices):
    moved = prices.reindex({"region": ["south"], "product": ["c", "b"]})
    assert moved.shape == (1, 2)
    np.testing.assert_array_equal(moved["price"], prices["price"][[1]][:, [2, 1]])


def test_an_empty_label_set_gives_an_empty_axis(prices):
    assert prices.reindex(product=[]).shape == (2, 0)


# -- labels the array does not have ---------------------------------------------


def test_new_labels_are_filled_and_marked_absent(prices):
    wider = prices.reindex(product=["a", "c", "d"])
    assert wider.dims["product"].tolist() == ["a", "c", "d"]
    assert np.isnan(wider["price"][:, 2]).all()
    assert wider.mask[:, 2].tolist() == [False, False]
    assert wider.mask[:, :2].tolist() == prices.mask[:, [0, 2]].tolist(), "the rest is unchanged"


def test_the_fill_keeps_the_dtype_of_each_variable(prices):
    wider = prices.reindex(product=["a", "d"])
    assert wider.dtypes["units"] == np.dtype("int64"), "an int column is not widened to float"
    assert wider["units"][:, 1].tolist() == [0, 0], "the mask is what says the cell is empty"
    assert not wider.mask[:, 1].any()


def test_fill_value_for_every_variable(prices):
    wider = prices.reindex(product=["a", "d"], fill_value=-1)
    assert wider["price"][:, 1].tolist() == [-1.0, -1.0]
    assert wider["units"][:, 1].tolist() == [-1, -1]


def test_fill_value_per_variable(prices):
    wider = prices.reindex(product=["a", "d"], fill_value={"price": -1.0})
    assert wider["price"][:, 1].tolist() == [-1.0, -1.0]
    assert wider["units"][:, 1].tolist() == [0, 0], "unnamed variables take the dtype default"


def test_fill_value_for_an_unknown_variable_raises(prices):
    with pytest.raises(KeyError, match="unknown variables"):
        prices.reindex(product=["a", "d"], fill_value={"nope": 0})


def test_an_existing_mask_is_kept(prices):
    wider = prices.reindex(product=["a", "b", "c", "d"])
    assert wider.mask.tolist() == [
        [True, True, True, False],
        [True, True, False, False],
    ]


def test_missing_labels_on_two_dimensions_intersect(prices):
    wider = prices.reindex(region=["north", "east"], product=["a", "d"])
    assert wider.mask.tolist() == [[True, False], [False, False]]


def test_density_reflects_what_was_added(prices):
    assert prices.reindex(product=["a", "b", "c", "d"]).density == 5 / 8


# -- what travels with the values -----------------------------------------------


def test_attributes_survive(prices):
    assert prices.reindex(product=["a"]).attrs == {"source": "fixture"}


def test_coordinates_are_gathered_and_filled():
    dates = np.array([["2024-01-01", "2024-01-02"], ["2024-02-01", "2024-02-02"]], "datetime64[D]")
    array = LabelledArray(
        {"v": np.arange(4.0).reshape(2, 2)},
        {"g": ["a", "b"], "i": [0, 1]},
        coords={"date": dates},
    )
    moved = array.reindex(g=["b", "c"])
    np.testing.assert_array_equal(moved.coords["date"][0], dates[1])
    assert np.isnat(moved.coords["date"][1]).all(), "a coordinate with no cell is null"
    assert moved.mask.tolist() == [[True, True], [False, False]]


def test_a_natural_index_can_be_reindexed():
    array = LabelledArray({"v": np.arange(3.0)})
    grown = array.reindex(dim_0=[0, 1, 2, 3])
    assert grown.shape == (4,)
    assert grown.dims.is_natural("dim_0")


def test_the_subclass_is_kept(prices):
    class Custom(LabelledArray):
        pass

    custom = Custom(dict(prices.data), prices.dims)
    assert isinstance(custom.reindex(product=["a"]), Custom)


def test_reindex_runs_on_every_backend(prices, backend):
    moved = prices.to_backend(backend).reindex(product=["a", "d"])
    values = get_numpy(moved, "price", backend)
    assert values[:, 0].tolist() == [0.0, 3.0]
    assert np.isnan(values[:, 1]).all()


def get_numpy(array, name, backend):
    """Return one variable as numpy, whatever backend it is held on."""
    return backend.to_numpy(array[name])


# -- errors ---------------------------------------------------------------------


def test_unknown_dimension_raises(prices):
    with pytest.raises(DimensionError, match="unknown dimensions"):
        prices.reindex(nope=["a"])


def test_duplicate_labels_raise(prices):
    with pytest.raises(DimensionError, match="must be unique"):
        prices.reindex(product=["a", "a"])


def test_null_labels_raise(prices):
    with pytest.raises(DimensionError, match="must not be null"):
        prices.reindex(product=[np.nan])


# -- reindex_like ---------------------------------------------------------------


def test_reindex_like_takes_the_other_array_labels(prices):
    target = LabelledArray(
        {"price": np.zeros((1, 2))}, {"region": ["south"], "product": ["c", "a"]}
    )
    moved = prices.reindex_like(target)
    assert moved.dims["region"].tolist() == ["south"]
    assert moved.dims["product"].tolist() == ["c", "a"]


def test_reindex_like_ignores_dimensions_this_array_does_not_have(prices):
    target = LabelledArray({"v": np.zeros(2)}, {"other": [1, 2]})
    assert prices.reindex_like(target).equals(prices)


def test_reindex_like_fills(prices):
    target = LabelledArray({"v": np.zeros(2)}, {"product": ["a", "z"]})
    assert prices.reindex_like(target, fill_value=0.0)["price"][:, 1].tolist() == [0.0, 0.0]


# -- fillna ---------------------------------------------------------------------


def test_fillna_fills_the_unpopulated_cells_and_drops_the_mask(prices):
    filled = prices.fillna(0)
    assert filled.mask is None
    assert filled["price"][1, 2] == 0.0
    assert filled["price"][0, 0] == prices["price"][0, 0]


def test_fillna_fills_nulls_that_were_never_masked():
    array = LabelledArray({"v": np.array([1.0, np.nan, 3.0])}, {"i": [0, 1, 2]})
    assert array.fillna(-1)["v"].tolist() == [1.0, -1.0, 3.0]


def test_fillna_per_variable_keeps_the_mask(prices):
    filled = prices.fillna({"price": 0.0})
    assert filled.mask is not None, "units still has an empty cell"
    assert filled["price"][1, 2] == 0.0
    assert filled["units"][1, 2] == prices["units"][1, 2]


def test_fillna_keeps_the_dtype(prices):
    assert prices.fillna(0).dtypes["units"] == np.dtype("int64")


def test_fillna_emits_every_cell_to_a_frame(prices):
    assert len(prices.fillna(0).to_dataframe()) == prices.size
    assert len(prices.to_dataframe()) == prices.size - 1


def test_fillna_on_every_backend(prices, backend):
    filled = prices.to_backend(backend).fillna(0)
    assert filled.mask is None
    assert backend.to_numpy(filled["price"])[1, 2] == 0.0


def test_fillna_rejects_an_unknown_variable(prices):
    with pytest.raises(KeyError, match="unknown variables"):
        prices.fillna({"nope": 0})
