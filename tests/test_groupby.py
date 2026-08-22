"""Grouped reductions along one dimension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labelled_array import DimensionError, GroupBy, LabelledArray

SECTORS = ["tech", "tech", "banks", "banks"]


@pytest.fixture
def book():
    """Four assets over three dates, with a sector per asset."""
    return LabelledArray(
        {"pnl": np.arange(12.0).reshape(4, 3)},
        {"asset": ["a", "b", "c", "d"], "date": [1, 2, 3]},
        coords={"sector": np.array([[s] * 3 for s in SECTORS], dtype=object)},
        attrs={"source": "fixture"},
    )


@pytest.fixture
def frame(book):
    """The same numbers as rows, for pandas to answer independently."""
    return pd.DataFrame(
        {
            "sector": np.repeat(SECTORS, 3),
            "date": list(book.dims["date"]) * 4,
            "pnl": book["pnl"].reshape(-1),
        }
    )


# -- what the groups are ---------------------------------------------------------


def test_group_values_are_given_per_label(book):
    grouped = book.groupby(SECTORS, dim="asset")
    assert len(grouped) == 2
    assert grouped.labels.tolist() == ["banks", "tech"], "sorted, as labels always are"
    assert grouped.dim == "asset"


def test_groups_map_each_value_to_its_positions(book):
    groups = book.groupby(SECTORS, dim="asset").groups
    assert groups["tech"].tolist() == [0, 1]
    assert groups["banks"].tolist() == [2, 3]


def test_iterating_yields_the_sub_arrays(book):
    pieces = dict(book.groupby(SECTORS, dim="asset"))
    assert pieces["tech"].dims["asset"].tolist() == ["a", "b"]
    np.testing.assert_array_equal(pieces["banks"]["pnl"], book["pnl"][2:])


def test_the_class_is_importable_from_the_package(book):
    assert isinstance(book.groupby(SECTORS, dim="asset"), GroupBy)


def test_repr_names_the_dimension_and_the_count(book):
    assert repr(book.groupby(SECTORS, dim="asset")) == (
        "GroupBy(LabelledArray over 'asset', 2 groups)"
    )


# -- the reductions --------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_grouped_reductions_agree_with_pandas(book, frame, op):
    grouped = getattr(book.groupby(SECTORS, dim="asset"), op)()
    expected = getattr(frame.groupby(["sector", "date"])["pnl"], op)().unstack()
    assert grouped.dims["asset"].tolist() == expected.index.tolist()
    np.testing.assert_allclose(grouped["pnl"], expected.to_numpy())


def test_the_grouping_dimension_keeps_its_name_and_gains_the_group_labels(book):
    grouped = book.groupby(SECTORS, dim="asset").sum()
    assert grouped.dims.names == ("asset", "date")
    assert grouped.dims["asset"].tolist() == ["banks", "tech"]
    assert grouped.shape == (2, 3)


def test_the_other_dimensions_are_untouched(book):
    assert book.groupby(SECTORS, dim="asset").mean().dims["date"].tolist() == [1, 2, 3]


def test_grouping_by_a_coordinate_finds_its_dimension(book):
    by_coord = book.groupby("sector").sum()
    by_array = book.groupby(SECTORS, dim="asset").sum()
    assert by_coord.dims["asset"].tolist() == by_array.dims["asset"].tolist()
    np.testing.assert_array_equal(by_coord["pnl"], by_array["pnl"])


def test_the_coordinate_grouped_by_survives(book):
    grouped = book.groupby("sector").sum()
    assert grouped.coords["sector"][:, 0].tolist() == ["banks", "tech"]


def test_naming_the_dimension_as_well_is_allowed(book):
    assert book.groupby("sector", dim="asset").sum().shape == (2, 3)


def test_absent_cells_do_not_contribute():
    array = LabelledArray(
        {"w": np.arange(4.0).reshape(4, 1)},
        {"asset": ["a", "b", "c", "d"], "t": [0]},
        mask=np.array([[True], [False], [False], [False]]),
    )
    grouped = array.groupby(SECTORS, dim="asset").sum()
    assert grouped["w"][:, 0].tolist()[1] == 0.0, "only 'a' is populated in tech"
    assert grouped.mask.tolist() == [[False], [True]]
    assert np.isnan(grouped["w"][0, 0]), "banks has nothing behind it"


def test_a_single_group_is_a_reduction(book):
    grouped = book.groupby(["all"] * 4, dim="asset").sum()
    assert grouped.shape == (1, 3)
    np.testing.assert_allclose(grouped["pnl"][0], book["pnl"].sum(axis=0))


def test_one_group_per_label_changes_nothing_but_the_labels(book):
    grouped = book.groupby(["w", "x", "y", "z"], dim="asset").sum()
    np.testing.assert_allclose(grouped["pnl"], book["pnl"])
    assert grouped.dims["asset"].tolist() == ["w", "x", "y", "z"]


def test_attributes_and_subclass_survive(book):
    class Custom(LabelledArray):
        pass

    custom = Custom(dict(book.data), book.dims, attrs={"k": 1})
    grouped = custom.groupby(SECTORS, dim="asset").sum()
    assert isinstance(grouped, Custom)
    assert grouped.attrs == {"k": 1}


def test_grouping_runs_on_every_backend(book, backend):
    grouped = book.to_backend(backend).groupby(SECTORS, dim="asset").mean()
    np.testing.assert_allclose(
        backend.to_numpy(grouped["pnl"]),
        book.groupby(SECTORS, dim="asset").mean()["pnl"],
        rtol=1e-6,
    )


def test_grouped_dates_can_be_compared_but_not_summed():
    array = LabelledArray(
        {"t": np.array(["2024-01-02", "2024-01-01"], "datetime64[D]")}, {"i": [0, 1]}
    )
    assert array.groupby(["g", "g"], dim="i").min()["t"].tolist() == [
        np.datetime64("2024-01-01").astype("O")
    ]
    with pytest.raises(TypeError, match="not added"):
        array.groupby(["g", "g"], dim="i").sum()


# -- errors ----------------------------------------------------------------------


def test_grouping_by_a_dimension_is_refused(book):
    with pytest.raises(DimensionError, match="its labels are unique"):
        book.groupby("asset")


def test_grouping_by_an_unknown_name_is_refused(book):
    with pytest.raises(DimensionError, match="unknown coordinate"):
        book.groupby("nope")


def test_a_coordinate_varying_along_two_dimensions_is_refused(book):
    array = book._replace(coords={"cell": np.arange(12).reshape(4, 3)})
    with pytest.raises(DimensionError, match="varies along"):
        array.groupby("cell")


def test_a_constant_coordinate_is_refused(book):
    array = book._replace(coords={"flat": np.zeros((4, 3))})
    with pytest.raises(DimensionError, match="varies along no dimension"):
        array.groupby("flat")


def test_the_wrong_dimension_for_a_coordinate_is_refused(book):
    with pytest.raises(DimensionError, match="varies along 'asset', not along 'date'"):
        book.groupby("sector", dim="date")


def test_an_array_of_group_values_needs_a_dimension(book):
    with pytest.raises(DimensionError, match="dim must be given"):
        book.groupby(SECTORS)


def test_the_wrong_number_of_group_values_is_refused(book):
    with pytest.raises(DimensionError, match="got 2 group values"):
        book.groupby(["tech", "banks"], dim="asset")


def test_an_unknown_dimension_is_refused(book):
    with pytest.raises(DimensionError, match="unknown dimensions"):
        book.groupby(SECTORS, dim="nope")


def test_an_unknown_op_is_refused(book):
    with pytest.raises(ValueError, match="op must be one of"):
        book.groupby(SECTORS, dim="asset").reduce("median")
