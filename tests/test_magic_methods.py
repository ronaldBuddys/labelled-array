"""Tests for the magic methods of :class:`LabelledArray`."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from numpy import array  # noqa: F401  (needed to eval reprs)

from labelled_array import DimensionError, Dimensions, LabelledArray  # noqa: F401


# -- mapping-like access --------------------------------------------------------


def test_getitem_returns_a_variable(simple):
    np.testing.assert_array_equal(simple["w"], np.arange(6).reshape(2, 3))


def test_getitem_unknown_variable_raises(simple):
    with pytest.raises(KeyError, match="unknown variable 'zz'"):
        simple["zz"]


def test_getitem_list_returns_a_subset(simple):
    subset = simple[["x"]]
    assert isinstance(subset, LabelledArray)
    assert subset.variables == ("x",)
    assert subset.dims == simple.dims
    assert subset.mask.tolist() == simple.mask.tolist()


def test_getitem_list_preserves_the_requested_order(simple):
    assert simple[["x", "w"]].variables == ("x", "w")


def test_getitem_list_with_unknown_names_raises(simple):
    with pytest.raises(KeyError, match="unknown variables"):
        simple[["w", "zz"]]


def test_getitem_rejects_other_keys(simple):
    with pytest.raises(TypeError, match="variable name, a list of names, or a mapping"):
        simple[0]


def test_setitem_adds_a_variable(simple):
    simple["y"] = np.ones((2, 3))
    assert simple.variables == ("w", "x", "y")


def test_setitem_checks_the_shape(simple):
    with pytest.raises(DimensionError, match="expected"):
        simple["y"] = np.ones((3, 3))


def test_setitem_rejects_dimension_and_coord_names(simple):
    with pytest.raises(DimensionError, match="must be unique"):
        simple["a"] = np.ones((2, 3))


def test_setitem_enforces_subclass_validation():
    """Setting a variable is validated exactly as assign is."""

    class OnlyKnown(LabelledArray):
        def _validate(self):
            super()._validate()
            if set(self.variables) - {"w", "x"}:
                raise DimensionError("unexpected variable")

    la = OnlyKnown({"w": np.zeros(2)}, {"a": [1, 2]})
    with pytest.raises(DimensionError, match="unexpected variable"):
        la["surprise"] = np.zeros(2)
    assert la.variables == ("w",), "a rejected assignment leaves the array unchanged"


def test_setitem_requires_a_string_name(simple):
    with pytest.raises(TypeError, match="must be str"):
        simple[0] = np.ones((2, 3))


def test_delitem(simple):
    del simple["w"]
    assert simple.variables == ("x",)
    with pytest.raises(KeyError, match="unknown variable"):
        del simple["zz"]


def test_contains_iter_and_len_are_over_variables(simple):
    assert "w" in simple
    assert "a" not in simple, "dimensions are not variables"
    assert list(simple) == ["w", "x"]
    assert len(simple) == 2, "len counts variables, size counts cells"
    assert simple.size == 6


# -- equality and hashing -------------------------------------------------------


def test_eq_is_structural(simple):
    assert simple == simple.copy()
    assert simple != simple.drop_vars("w")
    assert (simple == "not an array") is False


def test_unhashable(simple):
    with pytest.raises(TypeError):
        hash(simple)


# -- numpy interoperability -----------------------------------------------------


def test_array_protocol(simple):
    np.testing.assert_array_equal(np.asarray(simple[["w"]]), simple["w"])
    assert np.asarray(simple[["w"]], dtype="float32").dtype == np.dtype("float32")


def test_array_protocol_requires_a_single_variable(simple):
    with pytest.raises(ValueError, match="single variable"):
        np.asarray(simple)


def test_array_protocol_honours_copy(simple):
    single = simple[["w"]]
    np.array(single, copy=True)[0, 0] = 99
    assert single["w"][0, 0] == 0


def test_array_protocol_on_other_backends(simple, backend):
    np.testing.assert_array_equal(np.asarray(simple[["x"]].to_backend(backend)), simple["x"])


# -- copying and pickling -------------------------------------------------------


def test_copy_module_shallow_shares_arrays(simple):
    copy.copy(simple)["w"][0, 0] = 99
    assert simple["w"][0, 0] == 99


def test_copy_module_deep_is_independent(simple):
    copy.deepcopy(simple)["w"][0, 0] = 99
    assert simple["w"][0, 0] == 0


def test_pickle_round_trip(simple):
    restored = pickle.loads(pickle.dumps(simple))
    assert restored.equals(simple)
    assert restored.attrs == simple.attrs
    assert restored.backend.name == simple.backend.name


def test_pickle_round_trip_with_coords_and_backend(simple, backend):
    la = simple[["x"]].to_backend(backend)._replace(coords={"d": np.arange(6).reshape(2, 3)})
    restored = pickle.loads(pickle.dumps(la))
    assert restored.equals(la)
    assert restored.backend.name == backend.name


def test_dimensions_pickle_round_trip(simple):
    assert pickle.loads(pickle.dumps(simple.dims)) == simple.dims


# -- repr and str ---------------------------------------------------------------


def test_repr_round_trips(simple):
    assert eval(repr(simple)) == simple


def test_repr_round_trips_with_coords_and_natural_dims():
    la = LabelledArray(
        {"w": np.arange(4).reshape(2, 2)},
        Dimensions({"a": ["p", "q"], "date_idx": [0, 1]}, natural=["date_idx"]),
        mask=np.array([[True, False], [True, True]]),
        coords={"date": np.arange(4).reshape(2, 2).astype("datetime64[D]")},
        attrs={"unit": "m"},
    )
    assert eval(repr(la)) == la


def test_repr_round_trips_on_other_backends(simple, backend):
    la = simple[["x"]].to_backend(backend)
    restored = eval(repr(la))
    assert restored.equals(la)


def test_repr_records_a_non_default_backend(simple):
    assert "backend='torch'" in repr(simple[["x"]].to_backend("torch"))
    assert "backend=" not in repr(simple)


def test_repr_of_a_large_array_is_a_summary_not_a_broken_constructor():
    la = LabelledArray({"w": np.arange(2000)})
    text = repr(la)
    assert "too large for an eval-able repr" in text
    with pytest.raises(SyntaxError):
        eval(text)


def test_to_repr_full_is_eval_able_for_a_large_array():
    la = LabelledArray({"w": np.arange(2000)})
    assert eval(la.to_repr(full=True)) == la


def test_to_repr_not_full_is_summarised():
    assert "..." in LabelledArray({"w": np.arange(2000)}).to_repr(full=False)


def test_repr_threshold_is_a_class_attribute():
    class Tiny(LabelledArray):
        _repr_full_threshold = 2

    assert "too large" in repr(Tiny({"w": np.arange(3)}))
    assert "too large" not in repr(Tiny({"w": np.arange(2)}))


def test_str_summarises(simple):
    text = str(simple)
    assert "2 variable(s) over 2 dimension(s)" in text
    assert "shape (2, 3)" in text
    assert "backend numpy" in text
    assert "w int64" in text
    assert "5 / 6 cells populated" in text
    assert "83.3% density" in text
    assert "source" in text


def test_str_without_mask_or_attrs():
    text = str(LabelledArray({"w": np.zeros(2)}, {"a": [1, 2]}))
    assert "Mask" not in text and "Attrs" not in text


def test_str_lists_coordinates(simple):
    assert "Coordinates:" in str(simple._replace(coords={"d": np.zeros((2, 3))}))


def test_repr_names_the_subclass():
    class Climate(LabelledArray):
        pass

    assert repr(Climate({"t": np.zeros(2)}, {"a": [1, 2]})).startswith("Climate(")


# -- arithmetic -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda la: la + 1, lambda w: w + 1),
        (lambda la: 1 + la, lambda w: 1 + w),
        (lambda la: la - 1, lambda w: w - 1),
        (lambda la: 1 - la, lambda w: 1 - w),
        (lambda la: la * 2, lambda w: w * 2),
        (lambda la: 2 * la, lambda w: 2 * w),
        (lambda la: la / 2, lambda w: w / 2),
        (lambda la: 2 / (la + 1), lambda w: 2 / (w + 1)),
        (lambda la: la // 2, lambda w: w // 2),
        (lambda la: la % 2, lambda w: w % 2),
        (lambda la: la**2, lambda w: w**2),
        (lambda la: 2 ** (la / 6), lambda w: 2 ** (w / 6)),
        (lambda la: -la, lambda w: -w),
        (lambda la: +la, lambda w: +w),
        (lambda la: abs(-la), lambda w: abs(-w)),
    ],
)
def test_scalar_arithmetic(simple, operation, expected):
    result = operation(simple[["x"]])
    np.testing.assert_allclose(result["x"], expected(simple["x"]))
    assert result.dims == simple.dims


def test_arithmetic_between_labelled_arrays(simple):
    doubled = simple + simple
    np.testing.assert_array_equal(doubled["w"], simple["w"] * 2)
    np.testing.assert_array_equal(doubled["x"], simple["x"] * 2)


def test_arithmetic_preserves_attrs_and_coords(simple):
    la = simple._replace(coords={"d": np.zeros((2, 3))})
    result = la + 1
    assert result.attrs == la.attrs
    assert list(result.coords) == ["d"]


def test_arithmetic_combines_masks(simple):
    other = simple._replace(mask=np.array([[True, False, True], [True, True, True]]))
    assert (simple + other).mask.tolist() == [[True, False, False], [True, True, True]]
    assert (simple + simple._replace(mask=None)).mask.tolist() == simple.mask.tolist()


def test_arithmetic_requires_matching_dimensions(simple):
    with pytest.raises(ValueError, match="different dimensions"):
        simple + simple.isel(a=[0])


def test_arithmetic_requires_matching_variables(simple):
    with pytest.raises(ValueError, match="different variables"):
        simple + simple.drop_vars("w")


def test_arithmetic_with_an_array(simple):
    np.testing.assert_array_equal((simple + np.ones((2, 3)))["w"], simple["w"] + 1)


def test_arithmetic_on_other_backends(simple, backend):
    result = simple[["x"]].to_backend(backend) * 2
    assert result.backend.name == backend.name
    np.testing.assert_allclose(backend.to_numpy(result["x"]), simple["x"] * 2)


def test_arithmetic_operators_are_real_methods():
    """They are written out, not generated, so tooling can find them."""
    for name in ("__add__", "__radd__", "__floordiv__", "__rmod__", "__abs__"):
        assert name in vars(LabelledArray) or name in vars(LabelledArray.__mro__[1])
