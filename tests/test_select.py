"""Label-based selection."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import assert_frames_equal, build_frame, n_rows, sort_frame
from labelled_array import DimensionError, LabelledArray, LabelNotFoundError


@pytest.fixture
def cube():
    """A 2 x 3 x 2 array with distinct labels per dimension."""
    return LabelledArray(
        {"w": np.arange(12).reshape(2, 3, 2), "x": np.arange(12.0).reshape(2, 3, 2)},
        {"a": [1, 2], "b": ["x", "y", "z"], "flag": [False, True]},
        mask=np.ones((2, 3, 2), dtype=bool),
    )


# -- dimensions are kept --------------------------------------------------------


def test_a_scalar_label_keeps_the_dimension(cube):
    selected = cube.select({"a": 1})
    assert selected.shape == (1, 3, 2)
    assert selected.ndim == cube.ndim
    assert selected.dims["a"].tolist() == [1]


def test_drop_removes_the_selected_size_one_dimensions(cube):
    selected = cube.select({"a": 1}, drop=True)
    assert selected.dims.names == ("b", "flag")
    assert selected.shape == (3, 2)
    np.testing.assert_array_equal(selected["w"], cube["w"][0])


def test_drop_only_removes_dimensions_that_were_selected(cube):
    single = cube.select({"b": "x"})
    assert single.select({"a": [1, 2]}, drop=True).dims.names == ("a", "b", "flag")


def test_drop_removes_several_dimensions_at_once(cube):
    assert cube.select({"a": 1, "b": "x"}, drop=True).dims.names == ("flag",)


def test_selecting_a_list_of_one_still_keeps_the_dimension(cube):
    assert cube.select({"a": [1]}).shape == (1, 3, 2)


# -- label containers and ordering ----------------------------------------------


@pytest.mark.parametrize("labels", [["x", "z"], ("x", "z"), np.array(["x", "z"])])
def test_label_containers(cube, labels):
    assert cube.select({"b": labels}).dims["b"].tolist() == ["x", "z"]


def test_the_requested_order_is_preserved(cube):
    selected = cube.select({"b": ["z", "x"]})
    assert selected.dims["b"].tolist() == ["z", "x"]
    np.testing.assert_array_equal(selected["w"], cube["w"][:, [2, 0]])


def test_selection_can_reorder_a_whole_dimension(cube):
    reversed_ = cube.select({"b": ["z", "y", "x"]})
    assert reversed_.shape == cube.shape
    np.testing.assert_array_equal(reversed_["w"], cube["w"][:, ::-1])


def test_several_dimensions_in_one_call(cube):
    selected = cube.select({"a": [2], "b": ["y", "x"], "flag": True})
    assert selected.shape == (1, 2, 1)
    np.testing.assert_array_equal(selected["w"], cube["w"][1:, [1, 0]][:, :, 1:])


def test_chained_selections_match_a_single_call(cube):
    chained = cube.select({"a": [2]}).select({"b": ["y"]})
    assert chained.equals(cube.select({"a": [2], "b": ["y"]}))


def test_selection_is_independent_of_dimension_order_in_the_mapping(cube):
    assert cube.select({"b": ["y"], "a": [2]}).equals(
        cube.select({"a": [2], "b": ["y"]})
    )


def test_empty_selection(cube):
    selected = cube.select({"b": []})
    assert selected.shape == (2, 0, 2)
    assert selected.size == 0
    assert selected.dims["b"].tolist() == []


def test_mapping_and_keyword_forms_agree(cube):
    assert cube.select({"a": 1}).equals(cube.select(a=1))
    assert cube.select({"a": 1}, drop=True).equals(cube.select(drop=True, a=1))


def test_getitem_with_a_mapping_selects(cube):
    assert cube[{"a": 1}].equals(cube.select({"a": 1}))


def test_selecting_nothing_returns_an_equal_array(cube):
    assert cube.select().equals(cube)


# -- strictness -----------------------------------------------------------------


def test_missing_labels_raise_by_default(cube):
    with pytest.raises(LabelNotFoundError) as excinfo:
        cube.select({"b": ["x", "nope"]})
    assert excinfo.value.dim == "b"
    assert excinfo.value.missing == ["nope"]
    assert "'b'" in str(excinfo.value) and "nope" in str(excinfo.value)


def test_missing_labels_can_be_ignored(cube):
    selected = cube.select({"b": ["z", "nope", "x"]}, missing="ignore")
    assert selected.dims["b"].tolist() == ["z", "x"]


def test_ignoring_every_label_gives_an_empty_dimension(cube):
    assert cube.select({"b": ["nope"]}, missing="ignore").shape == (2, 0, 2)


def test_an_invalid_missing_mode_raises(cube):
    with pytest.raises(ValueError, match="missing must be one of"):
        cube.select({"b": ["x"]}, missing="skip")


def test_an_unknown_dimension_raises(cube):
    with pytest.raises(DimensionError, match="unknown dimensions: \\['zz'\\]"):
        cube.select({"zz": [1]})


def test_a_repeated_label_raises(cube):
    with pytest.raises(
        DimensionError, match="requested more than once for dimension 'b'"
    ):
        cube.select({"b": ["x", "x"]})


def test_a_label_of_the_wrong_type_is_simply_absent(cube):
    with pytest.raises(LabelNotFoundError):
        cube.select({"a": "x"})


def test_long_strings_are_not_truncated_to_match(cube):
    with pytest.raises(LabelNotFoundError):
        cube.select({"b": ["xyz"]})


# -- what comes along with the selection ----------------------------------------


def test_the_mask_is_selected_too(simple):
    assert simple.select({"b": ["z"]}).mask.tolist() == [[False], [True]]


def test_coordinates_are_selected_too(simple):
    la = simple._replace(coords={"d": np.arange(6).reshape(2, 3)})
    assert la.select({"b": ["z", "x"]}).coords["d"].tolist() == [[2, 0], [5, 3]]


def test_attributes_are_carried_through(simple):
    assert simple.select({"a": [1]}).attrs == simple.attrs


def test_every_variable_is_selected(cube):
    selected = cube.select({"a": 2})
    np.testing.assert_array_equal(selected["w"], cube["w"][1:])
    np.testing.assert_array_equal(selected["x"], cube["x"][1:])


def test_selection_works_with_no_variables(cube):
    empty = cube.drop_vars(["w", "x"])
    assert empty.select({"a": 1}).shape == (1, 3, 2)


# -- labels of other dtypes -----------------------------------------------------


def test_boolean_labels(cube):
    assert cube.select({"flag": True}).dims["flag"].tolist() == [True]


def test_datetime_labels():
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    la = LabelledArray({"w": np.arange(3.0)}, {"date": dates})
    assert la.select({"date": np.datetime64("2024-01-02")})["w"].tolist() == [1.0]
    assert la.select({"date": "2024-01-03"})["w"].tolist() == [2.0]


def test_float_labels():
    la = LabelledArray({"w": np.arange(3.0)}, {"a": [0.5, 1.5, 2.5]})
    assert la.select({"a": 1.5})["w"].tolist() == [1.0]


def test_object_labels():
    la = LabelledArray({"w": np.arange(2.0)}, {"a": np.array(["p", "q"], dtype=object)})
    assert la.select({"a": "q"})["w"].tolist() == [1.0]


# -- interaction with the rest of the package -----------------------------------


def test_select_on_every_backend(cube, backend):
    moved = cube[["x"]].to_backend(backend)
    selected = moved.select({"b": ["z", "x"], "a": 2})
    np.testing.assert_array_equal(
        backend.to_numpy(selected["x"]), cube["x"][1:, [2, 0]]
    )
    assert selected.backend.name == backend.name


def test_select_returns_the_subclass(cube):
    class Climate(LabelledArray):
        pass

    la = Climate(dict(cube.data), cube.dims)
    assert type(la.select({"a": 1})) is Climate
    assert type(la.select({"a": 1}, drop=True)) is Climate


def test_a_selection_still_round_trips(engine, sales_data):
    df = build_frame(engine, sales_data)
    la = LabelledArray.from_dataframe(df, dims=["region", "product"])
    out = la.select({"region": "north"}).to_dataframe(engine, index=False)
    expected = {
        name: [
            v
            for v, r in zip(values, sales_data["region"], strict=False)
            if r == "north"
        ]
        for name, values in sales_data.items()
    }
    assert_frames_equal(
        engine,
        sort_frame(engine, build_frame(engine, expected), ["region", "product"]),
        out,
    )


def test_selecting_a_natural_dimension_by_position(engine):
    data = {
        "A": [1, 1, 2],
        "date": np.array(["2024-01-01", "2024-01-02", "2024-02-01"], "datetime64[D]"),
        "w": [1.0, 2.0, 3.0],
    }
    la = LabelledArray.from_dataframe(
        build_frame(engine, data), cols=["w"], dims=["A", "date"], natural=["date"]
    )
    first = la.select({"date_idx": 0})
    assert first.shape == (2, 1)
    assert n_rows(engine, first.to_dataframe(engine, index=False)) == 2


def test_select_then_isel_agree(cube):
    assert cube.select({"b": ["z", "x"]}).equals(cube.isel({"b": [2, 0]}))
