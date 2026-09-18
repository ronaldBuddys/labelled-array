"""What an absent cell holds: unmistakable by default, and the caller's to choose."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import build_frame
from labelled_array import LabelledArray, default_fill_value

INTEGERS = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


@pytest.mark.parametrize("dtype", INTEGERS)
def test_an_integer_gets_the_extreme_of_its_range(dtype):
    limits = np.iinfo(dtype)
    expected = limits.min if np.dtype(dtype).kind == "i" else limits.max
    fill = default_fill_value(dtype)
    assert fill == expected
    assert isinstance(fill, int), "a python int, so every backend takes it"
    assert np.full(1, fill, dtype=dtype)[0] == expected, "and the dtype can hold it"


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex128"])
def test_a_float_gets_nan(dtype):
    assert np.isnan(default_fill_value(dtype))


@pytest.mark.parametrize("dtype", ["datetime64[D]", "datetime64[ns]", "timedelta64[s]"])
def test_a_time_gets_nat_in_its_own_unit(dtype):
    fill = default_fill_value(dtype)
    assert np.isnat(fill)
    assert np.datetime_data(fill.dtype) == np.datetime_data(np.dtype(dtype))


@pytest.mark.parametrize(
    "dtype, expected", [("bool", False), ("U3", ""), ("S3", ""), ("O", None)]
)
def test_dtypes_with_no_value_to_spare(dtype, expected):
    assert default_fill_value(dtype) == expected


@pytest.fixture
def sparse():
    return {
        "a": ["x", "x", "y"],
        "b": [1, 2, 1],
        "k": np.array([10, 20, 30], dtype="int32"),
        "u": np.array([1, 2, 3], dtype="uint8"),
    }


def test_from_dataframe_fills_with_the_default(engine, sparse):
    la = LabelledArray.from_dataframe(build_frame(engine, sparse), dims=["a", "b"])
    assert la["k"][1, 1] == np.iinfo("int32").min
    assert la["u"][1, 1] == 255
    assert la.dtypes == {"k": np.dtype("int32"), "u": np.dtype("uint8")}
    assert not la.mask[1, 1]


def test_the_default_is_what_shows_with_the_mask_off(engine, sparse):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sparse), dims=["a", "b"], mask=False
    )
    out = la.to_dataframe(engine, index=False)
    assert sorted(np.asarray(out["k"].to_numpy()).tolist()) == sorted(
        [10, 20, 30, np.iinfo("int32").min]
    ), "the invented row cannot be mistaken for a real one"


def test_fill_value_overrides_the_default(engine, sparse):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sparse), dims=["a", "b"], fill_value={"k": 0}
    )
    assert la["k"][1, 1] == 0
    assert la["u"][1, 1] == 255, "unnamed variables keep the default"


def test_the_round_trip_never_shows_the_fill(engine, sparse):
    la = LabelledArray.from_dataframe(build_frame(engine, sparse), dims=["a", "b"])
    out = la.to_dataframe(engine, index=False)
    assert np.asarray(out["k"].to_numpy()).tolist() == [10, 20, 30]


def test_every_operation_that_invents_a_cell_uses_the_default(backend):
    array = LabelledArray(
        {"k": np.arange(4).reshape(2, 2)},
        {"a": [1, 2], "b": ["x", "y"]},
        mask=np.array([[True, True], [False, False]]),
        backend=backend,
    )
    # JAX holds integers as int32 unless told otherwise, and the fill follows.
    absent = default_fill_value(array.dtypes["k"])
    assert absent == np.iinfo(array.dtypes["k"]).min
    as_numpy = backend.to_numpy
    assert as_numpy(array.reindex(a=[1, 3])["k"])[1].tolist() == [absent, absent]
    assert as_numpy(array.sum("b")["k"]).tolist() == [1, absent]
    assert as_numpy(array.rolling("a", 2).sum()["k"])[0].tolist() == [absent, absent]


def test_reductions_never_read_the_fill():
    # The fill is extreme, so a reduction that touched it would overflow or be
    # wildly off; absent cells contribute the operation's identity instead.
    array = LabelledArray(
        {"k": np.array([[5, np.iinfo("int64").min]])},
        {"a": [1], "b": ["x", "y"]},
        mask=np.array([[True, False]]),
    )
    assert array.sum()["k"] == 5
    assert array.min()["k"] == 5
    assert array.max()["k"] == 5
    assert array.mean()["k"] == 5.0
