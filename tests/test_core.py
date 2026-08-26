"""Tests for :class:`labelled_array.core.LabelledArray` construction and methods."""

from __future__ import annotations

import numpy as np
import pytest

from labelled_array import BackendDtypeError, DimensionError, Dimensions, LabelledArray

# -- construction ---------------------------------------------------------------


def test_construction_from_mapping(simple):
    assert simple.variables == ("w", "x")
    assert simple.shape == (2, 3)
    assert simple.ndim == 2
    assert simple.size == 6
    assert simple.dims.names == ("a", "b")
    assert simple.attrs == {"source": "fixture"}


def test_dims_default_to_a_natural_index():
    la = LabelledArray({"w": np.zeros((2, 3))})
    assert la.dims.names == ("dim_0", "dim_1")
    assert all(la.dims.is_natural(name) for name in la.dims.names)


def test_dims_accept_a_mapping_a_sequence_or_dimensions():
    array = np.zeros((2,))
    expected = Dimensions({"a": [1, 2]})
    assert LabelledArray({"w": array}, {"a": [1, 2]}).dims == expected
    assert LabelledArray({"w": array}, [("a", [1, 2])]).dims == expected
    assert LabelledArray({"w": array}, expected).dims == expected


def test_from_array_names_the_variable():
    la = LabelledArray.from_array(np.zeros((2, 2)), name="temperature")
    assert la.variables == ("temperature",)
    assert LabelledArray.from_array(np.zeros(2)).variables == ("values",)


def test_no_variables_requires_dims():
    assert LabelledArray({}, {"a": [1, 2]}).shape == (2,)
    with pytest.raises(DimensionError, match="dims must be given"):
        LabelledArray({})


def test_zero_dimensional_array():
    la = LabelledArray({"w": np.array(1.0)}, {})
    assert la.shape == () and la.size == 1


def test_data_must_be_a_mapping():
    with pytest.raises(TypeError, match="use LabelledArray.from_array"):
        LabelledArray(np.zeros(3))


def test_variable_names_must_be_str():
    with pytest.raises(TypeError, match="variable names must be str"):
        LabelledArray({0: np.zeros(3)}, {"a": [1, 2, 3]})


def test_shape_mismatch_raises():
    with pytest.raises(
        DimensionError, match=r"variable 'w' has shape \(3,\), expected \(2,\)"
    ):
        LabelledArray({"w": np.zeros(3)}, {"a": [1, 2]})


def test_names_are_unique_across_namespaces():
    with pytest.raises(DimensionError, match="must be unique"):
        LabelledArray({"a": np.zeros(2)}, {"a": [1, 2]})
    with pytest.raises(DimensionError, match="must be unique"):
        LabelledArray({"w": np.zeros(2)}, {"a": [1, 2]}, coords={"w": np.zeros(2)})


def test_mask_shape_is_checked():
    with pytest.raises(DimensionError, match="mask has shape"):
        LabelledArray({"w": np.zeros(2)}, {"a": [1, 2]}, mask=np.ones(3, dtype=bool))


def test_mask_is_read_only_and_decoupled_from_its_source():
    source = np.ones((2,), dtype=bool)
    la = LabelledArray({"w": np.zeros(2)}, {"a": [1, 2]}, mask=source)
    source[0] = False
    assert la.mask.tolist() == [True, True]
    with pytest.raises(ValueError):
        la.mask[0] = False


def test_coord_shape_is_checked():
    with pytest.raises(DimensionError, match="coordinate 'd' has shape"):
        LabelledArray({"w": np.zeros(2)}, {"a": [1, 2]}, coords={"d": np.zeros(3)})


def test_density_reflects_the_mask(simple):
    assert simple.density == pytest.approx(5 / 6)
    assert LabelledArray({"w": np.zeros(2)}, {"a": [1, 2]}).density == 1.0
    assert (
        LabelledArray({"w": np.zeros(0)}, {"a": []}, mask=np.zeros(0, bool)).density
        == 0.0
    )


# -- backend handling -----------------------------------------------------------


def test_backend_defaults_to_numpy(simple):
    assert simple.backend.name == "numpy"


def test_backend_is_inferred_from_the_data():
    import torch

    la = LabelledArray({"w": torch.zeros(2, 3)})
    assert la.backend.name == "torch"


def test_backend_argument_converts_every_variable():
    la = LabelledArray({"w": np.zeros((2, 3)), "x": np.ones((2, 3))}, backend="torch")
    assert la.backend.name == "torch"
    assert all(type(value).__name__ == "Tensor" for value in la.data.values())


def test_to_backend_round_trips(simple, backend):
    moved = simple.to_backend(backend)
    assert moved.backend.name == backend.name
    assert moved.to_backend("numpy").equals(simple)


def test_to_backend_rejects_unsupported_dtypes():
    la = LabelledArray({"ticker": np.array(["a", "b"])}, {"a": [1, 2]})
    with pytest.raises(BackendDtypeError, match="for 'ticker'"):
        la.to_backend("torch")


def test_dtypes_are_reported_as_numpy_dtypes(simple):
    assert simple.dtypes == {"w": np.dtype("int64"), "x": np.dtype("float64")}


# -- properties -----------------------------------------------------------------


def test_values_requires_a_single_variable(simple):
    with pytest.raises(ValueError, match="single variable"):
        _ = simple.values
    np.testing.assert_array_equal(simple[["w"]].values, simple["w"])


def test_data_and_coords_views_are_read_only(simple):
    with pytest.raises(TypeError):
        simple.data["w"] = np.zeros((2, 3))
    with pytest.raises(TypeError):
        simple.coords["d"] = np.zeros((2, 3))


def test_attrs_are_mutable_and_copied_from_the_source():
    attrs = {"a": 1}
    la = LabelledArray({"w": np.zeros(2)}, {"a": [1, 2]}, attrs=attrs)
    attrs["a"] = 2
    assert la.attrs == {"a": 1}
    la.attrs["b"] = 3
    assert la.attrs == {"a": 1, "b": 3}


# -- isel -----------------------------------------------------------------------


def test_isel_keeps_dimensions_by_default(simple):
    selected = simple.isel(a=0)
    assert selected.shape == (1, 3)
    assert selected.dims["a"].tolist() == [1]
    np.testing.assert_array_equal(selected["w"], simple["w"][:1])


def test_isel_drop_removes_selected_size_one_dimensions(simple):
    selected = simple.isel(a=0, drop=True)
    assert selected.dims.names == ("b",)
    assert selected.shape == (3,)
    np.testing.assert_array_equal(selected["w"], simple["w"][0])


def test_isel_drop_leaves_untouched_size_one_dimensions():
    la = LabelledArray({"w": np.zeros((1, 3))}, {"a": [1], "b": [1, 2, 3]})
    assert la.isel(b=[0], drop=True).dims.names == ("a",)


def test_isel_accepts_slices_lists_and_arrays(simple):
    assert simple.isel(b=slice(0, 2)).shape == (2, 2)
    assert simple.isel(b=[2, 0]).dims["b"].tolist() == ["z", "x"]
    assert simple.isel(b=np.array([1])).shape == (2, 1)


def test_isel_accepts_a_boolean_mask(simple):
    assert simple.isel(b=[True, False, True]).dims["b"].tolist() == ["x", "z"]


def test_isel_supports_negative_positions(simple):
    assert simple.isel(b=-1).dims["b"].tolist() == ["z"]


def test_isel_takes_mask_and_coords_along(simple):
    la = simple._replace(coords={"d": np.arange(6).reshape(2, 3)})
    selected = la.isel(b=[2])
    assert selected.mask.tolist() == [[False], [True]]
    assert selected.coords["d"].tolist() == [[2], [5]]


def test_isel_accepts_a_mapping_or_keywords(simple):
    assert simple.isel({"a": 0}).equals(simple.isel(a=0))


@pytest.mark.parametrize(
    ("indexers", "error", "match"),
    [
        ({"zz": 0}, DimensionError, "unknown dimensions"),
        ({"a": 5}, IndexError, "out of range"),
        ({"a": -5}, IndexError, "out of range"),
        ({"a": [[0]]}, DimensionError, "one-dimensional"),
        ({"a": 1.5}, DimensionError, "must be integers"),
        ({"a": [True]}, DimensionError, "boolean mask"),
    ],
)
def test_isel_invalid_inputs(simple, indexers, error, match):
    with pytest.raises(error, match=match):
        simple.isel(indexers)


def test_isel_empty_selection(simple):
    selected = simple.isel(a=[])
    assert selected.shape == (0, 3)
    assert selected.size == 0


def test_isel_on_every_backend(simple, backend):
    moved = simple[["x"]].to_backend(backend)
    selected = moved.isel(b=[2, 0])
    np.testing.assert_array_equal(
        backend.to_numpy(selected["x"]), simple["x"][:, [2, 0]]
    )


# -- rename, transpose, astype --------------------------------------------------


def test_rename_dimensions_variables_and_coords(simple):
    la = simple._replace(coords={"d": np.zeros((2, 3))})
    renamed = la.rename({"a": "A", "w": "W", "d": "D"})
    assert renamed.dims.names == ("A", "b")
    assert renamed.variables == ("W", "x")
    assert list(renamed.coords) == ["D"]


def test_rename_unknown_name_raises(simple):
    with pytest.raises(DimensionError, match="unknown names"):
        simple.rename(zz="A")


def test_transpose(simple):
    assert simple.transpose().dims.names == ("b", "a")
    assert simple.transpose("b", "a").shape == (3, 2)
    assert simple.transpose(["b", "a"]).shape == (3, 2)
    np.testing.assert_array_equal(simple.transpose()["w"], simple["w"].T)
    assert simple.transpose().mask.tolist() == simple.mask.T.tolist()


def test_transpose_round_trips(simple):
    assert simple.transpose().transpose().equals(simple)


def test_transpose_requires_a_permutation(simple):
    with pytest.raises(DimensionError, match="not a permutation"):
        simple.transpose("b")


def test_astype_all_variables(simple):
    assert simple.astype("float32").dtypes == {
        "w": np.dtype("float32"),
        "x": np.dtype("float32"),
    }


def test_astype_per_variable(simple):
    assert simple.astype({"w": "float64"}).dtypes["w"] == np.dtype("float64")


def test_astype_unknown_variable_raises(simple):
    with pytest.raises(KeyError, match="unknown variables"):
        simple.astype({"zz": "float64"})


# -- assign, drop_vars, copy ----------------------------------------------------


def test_assign_adds_and_replaces(simple):
    assigned = simple.assign(y=np.ones((2, 3)))
    assert assigned.variables == ("w", "x", "y")
    assert simple.variables == ("w", "x")
    np.testing.assert_array_equal(
        simple.assign(w=np.ones((2, 3)))["w"], np.ones((2, 3))
    )


def test_assign_checks_the_shape(simple):
    with pytest.raises(DimensionError, match="expected"):
        simple.assign(y=np.ones((3, 3)))


def test_drop_vars(simple):
    assert simple.drop_vars("w").variables == ("x",)
    assert simple.drop_vars(["w", "x"]).variables == ()
    with pytest.raises(KeyError, match="unknown variables"):
        simple.drop_vars("zz")


def test_copy_deep_is_independent(simple):
    copied = simple.copy()
    copied["w"][0, 0] = 99
    copied.attrs["source"] = "changed"
    assert simple["w"][0, 0] == 0
    assert simple.attrs["source"] == "fixture"


def test_copy_shallow_shares_arrays(simple):
    copied = simple.copy(deep=False)
    copied["w"][0, 0] = 99
    assert simple["w"][0, 0] == 99


# -- equality -------------------------------------------------------------------


def test_equals_is_structural(simple):
    assert simple.equals(simple.copy())
    assert not simple.equals(simple.drop_vars("w"))
    assert not simple.equals(simple.rename(a="A"))
    assert not simple.equals(simple.assign(w=np.ones((2, 3))))
    assert not simple.equals("not an array")


def test_equals_compares_masks(simple):
    assert not simple.equals(simple._replace(mask=None))
    assert not simple.equals(simple._replace(mask=np.ones((2, 3), dtype=bool)))


def test_equals_compares_coords(simple):
    with_coord = simple._replace(coords={"d": np.zeros((2, 3))})
    assert not simple.equals(with_coord)
    assert with_coord.equals(with_coord.copy())
    assert not with_coord.equals(with_coord._replace(coords={"d": np.ones((2, 3))}))


def test_equals_ignores_backend_and_attrs(simple, backend):
    assert simple.to_backend(backend).equals(simple)
    assert simple._replace(attrs={"other": 1}).equals(simple)


# -- subclassing ----------------------------------------------------------------


class FinancialArray(LabelledArray):
    """Test subclass requiring a date dimension."""

    _default_backend = "numpy"

    def _validate(self) -> None:
        super()._validate()
        if "date" not in self.dims:
            raise DimensionError("a FinancialArray needs a 'date' dimension")

    @property
    def dates(self):
        return self.dims["date"]


@pytest.fixture
def financial():
    return FinancialArray(
        {"price": np.arange(6, dtype=float).reshape(3, 2)},
        {
            "date": np.array(
                ["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]"
            ),
            "ticker": ["AAA", "BBB"],
        },
    )


def test_subclass_validation_is_enforced():
    with pytest.raises(DimensionError, match="needs a 'date' dimension"):
        FinancialArray({"price": np.zeros(2)}, {"a": [1, 2]})


@pytest.mark.parametrize(
    "operation",
    [
        lambda la: la.isel(ticker=0),
        lambda la: la.isel(ticker=0, drop=False),
        lambda la: la.transpose(),
        lambda la: la.astype("float32"),
        lambda la: la.copy(),
        lambda la: la.assign(volume=np.ones((3, 2))),
        lambda la: la.drop_vars("price"),
        lambda la: la + 1,
        lambda la: -la,
        lambda la: la[["price"]],
        lambda la: la.to_backend("numpy"),
    ],
)
def test_derived_objects_keep_the_subclass(financial, operation):
    result = operation(financial)
    assert type(result) is FinancialArray
    assert result.dates.size == 3


def test_subclass_specific_api(financial):
    assert financial.dates.dtype == np.dtype("datetime64[D]")


def test_subclasses_can_carry_their_own_constructor_state():
    """`_fields` is the hook that makes extra state survive derived operations."""

    class Priced(LabelledArray):
        def __init__(self, data, dims=None, currency="USD", **kwargs):
            self.currency = currency
            super().__init__(data, dims, **kwargs)

        def _fields(self):
            return {**super()._fields(), "currency": self.currency}

    la = Priced({"p": np.arange(4.0)}, {"a": [1, 2, 3, 4]}, currency="EUR")
    for operation in (
        lambda x: x.transpose(),
        lambda x: x.select({"a": [1]}),
        lambda x: x.isel(a=0),
        lambda x: x + 1,
        lambda x: x.copy(),
        lambda x: x.astype("float32"),
    ):
        derived = operation(la)
        assert type(derived) is Priced
        assert derived.currency == "EUR"


def test_variables_may_be_given_as_plain_sequences():
    """Anything no backend recognises is converted by the default backend."""
    la = LabelledArray({"w": [1.0, 2.0]}, {"a": [1, 2]})
    assert la.backend.name == "numpy"
    assert la["w"].tolist() == [1.0, 2.0]
