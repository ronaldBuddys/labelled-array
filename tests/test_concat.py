"""Joining arrays along an existing or a new dimension."""

from __future__ import annotations

import numpy as np
import pytest

from labelled_array import DimensionError, LabelledArray


@pytest.fixture
def jan():
    return LabelledArray(
        {"v": np.arange(4.0).reshape(2, 2)},
        {"day": [1, 2], "g": ["a", "b"]},
        attrs={"source": "fixture"},
    )


@pytest.fixture
def feb(jan):
    return LabelledArray(
        {"v": np.arange(4.0).reshape(2, 2) + 10}, {"day": [3, 4], "g": ["a", "b"]}
    )


# -- an existing dimension ------------------------------------------------------


def test_labels_and_values_are_joined(jan, feb):
    joined = LabelledArray.concat([jan, feb], dim="day")
    assert joined.shape == (4, 2)
    assert joined.dims["day"].tolist() == [1, 2, 3, 4]
    np.testing.assert_array_equal(joined["v"], np.concatenate([jan["v"], feb["v"]]))


def test_the_order_of_the_arrays_is_the_order_of_the_result(jan, feb):
    joined = LabelledArray.concat([feb, jan], dim="day")
    assert joined.dims["day"].tolist() == [3, 4, 1, 2]


def test_repeated_labels_are_refused(jan):
    with pytest.raises(DimensionError, match="must be unique"):
        LabelledArray.concat([jan, jan], dim="day")


def test_labels_can_be_given_explicitly(jan, feb):
    joined = LabelledArray.concat([jan, feb], dim="day", labels=list("wxyz"))
    assert joined.dims["day"].tolist() == ["w", "x", "y", "z"]


def test_the_wrong_number_of_labels_is_refused(jan, feb):
    with pytest.raises(DimensionError, match="got 2 labels for 4 entries"):
        LabelledArray.concat([jan, feb], dim="day", labels=[1, 2])


def test_a_natural_index_is_renumbered():
    first = LabelledArray({"v": np.arange(2.0)})
    joined = LabelledArray.concat([first, first], dim="dim_0")
    assert joined.dims["dim_0"].tolist() == [0, 1, 2, 3]
    assert joined.dims.is_natural("dim_0")


def test_concatenating_one_array_returns_it(jan):
    assert LabelledArray.concat([jan], dim="day") is jan


def test_transposed_arrays_are_put_in_the_first_array_order(jan, feb):
    joined = LabelledArray.concat([jan, feb.transpose("g", "day")], dim="day")
    assert joined.dims.names == ("day", "g")
    np.testing.assert_array_equal(joined["v"], np.concatenate([jan["v"], feb["v"]]))


# -- a new dimension ------------------------------------------------------------


def test_a_new_dimension_becomes_the_first_axis(jan, feb):
    stacked = LabelledArray.concat([jan, feb], dim="month")
    assert stacked.dims.names == ("month", "day", "g")
    assert stacked.shape == (2, 4, 2), (
        "the day labels are joined by the outer alignment"
    )
    assert stacked.dims["month"].tolist() == [0, 1]
    assert stacked.dims.is_natural("month")


def test_a_new_dimension_can_be_labelled(jan, feb):
    stacked = LabelledArray.concat(
        [jan, feb.reindex(day=[1, 2])], dim="month", labels=["jan", "feb"]
    )
    assert stacked.dims["month"].tolist() == ["jan", "feb"]
    assert not stacked.dims.is_natural("month")
    assert stacked.shape == (2, 2, 2)


def test_the_wrong_number_of_new_labels_is_refused(jan, feb):
    with pytest.raises(DimensionError, match="got 3 labels for 2 arrays"):
        LabelledArray.concat([jan, feb], dim="month", labels=[1, 2, 3])


def test_stacking_one_array_adds_the_axis(jan):
    assert LabelledArray.concat([jan], dim="month").shape == (1, 2, 2)


# -- the other dimensions -------------------------------------------------------


def test_the_other_dimensions_are_joined_outward_by_default(jan):
    other = LabelledArray(
        {"v": np.arange(2.0).reshape(1, 2)}, {"day": [3], "g": ["b", "c"]}
    )
    joined = LabelledArray.concat([jan, other], dim="day")
    assert joined.dims["g"].tolist() == ["a", "b", "c"]
    assert joined.mask.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, True, True],
    ]


def test_the_other_dimensions_can_be_joined_inward(jan):
    other = LabelledArray(
        {"v": np.arange(2.0).reshape(1, 2)}, {"day": [3], "g": ["b", "c"]}
    )
    joined = LabelledArray.concat([jan, other], dim="day", join="inner")
    assert joined.dims["g"].tolist() == ["b"]
    assert joined.mask is None


def test_an_exact_join_refuses_to_invent_cells(jan):
    other = LabelledArray(
        {"v": np.arange(2.0).reshape(1, 2)}, {"day": [3], "g": ["b", "c"]}
    )
    with pytest.raises(DimensionError, match="join='exact'"):
        LabelledArray.concat([jan, other], dim="day", join="exact")


def test_fill_value_reaches_the_cells_the_join_invents(jan):
    other = LabelledArray(
        {"v": np.arange(2.0).reshape(1, 2)}, {"day": [3], "g": ["b", "c"]}
    )
    joined = LabelledArray.concat([jan, other], dim="day", fill_value=-1.0)
    assert joined["v"][0, 2] == -1.0


# -- masks, coordinates and metadata --------------------------------------------


def test_masks_are_joined_and_absent_ones_count_as_populated(jan, feb):
    masked = jan._replace(mask=np.array([[True, False], [True, True]]))
    joined = LabelledArray.concat([masked, feb], dim="day")
    assert joined.mask.tolist() == [
        [True, False],
        [True, True],
        [True, True],
        [True, True],
    ]


def test_no_mask_at_all_stays_no_mask(jan, feb):
    assert LabelledArray.concat([jan, feb], dim="day").mask is None


def test_coordinates_are_joined():
    dates = np.array([["2024-01-01", "2024-01-02"]], "datetime64[D]")
    first = LabelledArray(
        {"v": np.zeros((1, 2))}, {"g": ["a"], "i": [0, 1]}, coords={"d": dates}
    )
    second = first._replace(dims=first.dims.replace("g", ["b"]))
    joined = LabelledArray.concat([first, second], dim="g")
    assert joined.coords["d"].shape == (2, 2)
    np.testing.assert_array_equal(joined.coords["d"][1], dates[0])


def test_attributes_and_subclass_come_from_the_first_array(jan, feb):
    class Custom(LabelledArray):
        pass

    custom = Custom(dict(jan.data), jan.dims, attrs={"k": 1})
    joined = Custom.concat([custom, feb], dim="day")
    assert isinstance(joined, Custom)
    assert joined.attrs == {"k": 1}


def test_concat_runs_on_every_backend(jan, feb, backend):
    joined = LabelledArray.concat(
        [jan.to_backend(backend), feb.to_backend(backend)], dim="day"
    )
    assert joined.backend.name == backend.name
    np.testing.assert_array_equal(
        backend.to_numpy(joined["v"]), np.concatenate([jan["v"], feb["v"]])
    )


def test_a_concatenated_array_still_round_trips(jan, feb):
    joined = LabelledArray.concat([jan, feb], dim="day")
    assert len(joined.to_dataframe()) == joined.size
    # str labels come back as object, as they do for any round trip.
    back = LabelledArray.from_dataframe(joined.to_dataframe())
    assert back.dims["day"].tolist() == joined.dims["day"].tolist()
    assert back.dims["g"].tolist() == joined.dims["g"].tolist()
    np.testing.assert_array_equal(back["v"], joined["v"])


# -- errors ---------------------------------------------------------------------


def test_no_arrays_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one array"):
        LabelledArray.concat([], dim="day")


def test_different_variables_are_refused(jan, feb):
    with pytest.raises(DimensionError, match="same variables"):
        LabelledArray.concat([jan, feb.assign(extra=np.zeros((2, 2)))], dim="day")


def test_different_coordinates_are_refused(jan, feb):
    with pytest.raises(DimensionError, match="same coordinates"):
        LabelledArray.concat(
            [jan, feb._replace(coords={"c": np.zeros((2, 2))})], dim="day"
        )


def test_different_backends_are_refused(jan, feb):
    with pytest.raises(DimensionError, match="one backend"):
        LabelledArray.concat([jan, feb.to_backend("torch")], dim="day")


def test_a_dimension_only_some_arrays_have_is_refused(jan):
    other = LabelledArray(
        {"v": np.zeros((2, 2, 1))}, {"day": [3, 4], "g": ["a", "b"], "x": [0]}
    )
    with pytest.raises(DimensionError, match="share their dimensions"):
        LabelledArray.concat([jan, other], dim="day")


def test_a_new_dimension_missing_from_only_some_arrays_is_refused(jan):
    with pytest.raises(DimensionError, match="some of the arrays but not all"):
        LabelledArray.concat([jan, jan.rename(day="month")], dim="day")
