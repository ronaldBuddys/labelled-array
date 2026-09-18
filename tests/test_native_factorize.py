"""An adapter's native factorize must reproduce the numpy reference exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from helpers import build_frame
from labelled_array import LabelledArray, _reshape
from labelled_array._utils import arrays_equal
from labelled_array.adapters import DataFrameAdapter, get_adapter

RNG = np.random.default_rng(0)

#: Strings that sort differently under a locale, a byte-wise C comparison that
#: stops at NUL, or UTF-16 -- none of which is the code point order wanted.
TRICKY = [
    "b",
    "",
    " ",
    "a",
    "B",
    "\u00e9",
    "e\u0301",
    "z\uffff",
    "\U0001f600",
    "\u00df",
    "ss",
]

STRINGS = {
    "ascii": ["north", "south", "north", "east"],
    "reverse_sorted": ["d", "c", "b", "a"],
    "already_sorted": ["a", "b", "c", "d"],
    "all_the_same": ["x", "x", "x"],
    "single_row": ["only"],
    "empty_string": ["", "a", "", "b"],
    "tricky_unicode": TRICKY + TRICKY[::-1],
    "high_cardinality": [f"label-{i}" for i in RNG.integers(0, 10_000, 30_000)],
}

DAYS = np.array(["2024-01-02", "2024-01-01", "2024-01-02"], dtype="datetime64[D]")

#: Columns both libraries can hold, none of which has a native path.
NOT_NATIVE = {
    "int": np.array([3, 1, 3], dtype="int64"),
    "uint": np.array([3, 1, 3], dtype="uint16"),
    "float": np.array([2.5, 1.5, 2.5]),
    "float_with_nan": np.array([2.5, np.nan, 2.5]),
    "bool": np.array([True, False, True]),
    "datetime_ns": DAYS.astype("datetime64[ns]"),
    "datetime_us": DAYS.astype("datetime64[us]"),
    "datetime_D": DAYS,
    "timedelta": np.array([2, 1, 2], dtype="timedelta64[ms]"),
    "strings_with_none": ["a", None, "b"],
}

#: Columns only one library can hold, which its native path must also refuse.
NOT_NATIVE_FRAMES = {
    "polars-date": pl.DataFrame({"a": pl.Series(DAYS).cast(pl.Date)}),
    "polars-categorical": pl.DataFrame(
        {"a": pl.Series(["b", "a"], dtype=pl.Categorical)}
    ),
    "polars-enum": pl.DataFrame(
        {"a": pl.Series(["b", "a"], dtype=pl.Enum(["b", "a"]))}
    ),
    "pandas-category": pd.DataFrame({"a": pd.Categorical(["b", "a"], ["b", "a"])}),
    "pandas-tz_aware": pd.DataFrame(
        {"a": pd.to_datetime(["2024-01-02", "2024-01-01"]).tz_localize("UTC")}
    ),
    "pandas-mixed_object": pd.DataFrame({"a": np.array([1, "a", 2.5], dtype=object)}),
    "pandas-strings_with_nan": pd.DataFrame(
        {"a": np.array(["a", np.nan, "b"], dtype=object)}
    ),
    "pandas-nullable_string": pd.DataFrame(
        {"a": pd.array(["a", pd.NA, "b"], dtype="string")}
    ),
}


def reference(engine: str, df) -> tuple[np.ndarray, np.ndarray]:
    """Return the numpy factorization of column ``a``."""
    return _reshape.factorize(get_adapter(engine).column_values(df, "a"))


def assert_same_factorization(got, want) -> None:
    """Assert two factorizations agree on everything an array is built from."""
    (labels, codes), (want_labels, want_codes) = got, want
    assert labels.dtype == want_labels.dtype
    assert [type(label) for label in labels] == [type(label) for label in want_labels]
    assert arrays_equal(labels, want_labels)  # a null label matches a null
    assert codes.dtype == np.intp
    np.testing.assert_array_equal(codes, want_codes)


@pytest.mark.parametrize("case", sorted(STRINGS))
def test_native_strings_match_the_numpy_reference(case):
    df = build_frame("polars", {"a": STRINGS[case]})
    native = get_adapter("polars")._native_factorize(df, "a")
    assert native is not None, "plain strings should take the native path"
    assert_same_factorization(native, reference("polars", df))


@pytest.mark.parametrize("case", sorted(STRINGS))
def test_pandas_has_no_native_path(case):
    # pd.factorize is barely faster than the reference and merges strings that
    # differ only after a NUL, so pandas columns always take the numpy path.
    df = build_frame("pandas", {"a": STRINGS[case]})
    adapter = get_adapter("pandas")
    assert adapter._native_factorize(df, "a") is None
    assert_same_factorization(adapter.factorize(df, "a"), reference("pandas", df))


@pytest.mark.parametrize("case", sorted(NOT_NATIVE))
def test_columns_with_no_native_path_return_none(engine, case):
    df = build_frame(engine, {"a": NOT_NATIVE[case]})
    assert get_adapter(engine)._native_factorize(df, "a") is None


@pytest.mark.parametrize("case", sorted(NOT_NATIVE_FRAMES))
def test_library_specific_columns_return_none(case):
    engine = case.split("-")[0]
    df = NOT_NATIVE_FRAMES[case]
    assert get_adapter(engine)._native_factorize(df, "a") is None


@pytest.mark.parametrize("case", sorted(NOT_NATIVE))
def test_factorize_falls_back_to_numpy(engine, case):
    df = build_frame(engine, {"a": NOT_NATIVE[case]})
    assert_same_factorization(
        get_adapter(engine).factorize(df, "a"), reference(engine, df)
    )


def test_pandas_keeps_nul_strings_apart():
    # pd.factorize hashes strings up to the first NUL, so these three would be
    # one label to it -- which is one reason the pandas adapter does not use it.
    df = pd.DataFrame({"a": ["a\x00b", "a\x00c", "a"]})
    labels, codes = get_adapter("pandas").factorize(df, "a")
    assert labels.tolist() == ["a", "a\x00b", "a\x00c"]
    assert codes.tolist() == [1, 2, 0]


def test_polars_keeps_nul_strings_apart():
    df = pl.DataFrame({"a": ["a\x00b", "a\x00c", "a"]})
    native = get_adapter("polars")._native_factorize(df, "a")
    assert native is not None
    assert_same_factorization(native, reference("polars", df))


def test_polars_factorizes_a_string_column_with_no_rows():
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.String)})
    labels, codes = get_adapter("polars").factorize(df, "a")
    assert labels.dtype == object and labels.shape == (0,)
    assert codes.dtype == np.intp and codes.shape == (0,)


def test_polars_string_dims_take_the_native_path(monkeypatch):
    def refuse(values):
        raise AssertionError("the numpy factorize should not have been reached")

    monkeypatch.setattr(_reshape, "factorize", refuse)
    df = pl.DataFrame(
        {"region": ["n", "s", "n"], "product": ["A", "A", "B"], "w": [1.0, 2.0, 3.0]}
    )
    la = LabelledArray.from_dataframe(df, dims=["region", "product"])
    assert la.dims["region"].tolist() == ["n", "s"]
    assert la.density == 0.75


class _DictAdapter(DataFrameAdapter):
    """An adapter from outside the package, overriding nothing it need not."""

    name = "dicts"
    library = "numpy"

    @classmethod
    def _instance_type(cls) -> type:
        return dict

    def n_rows(self, df) -> int:
        return len(next(iter(df.values())))

    def column_values(self, df, column) -> np.ndarray:
        return np.asarray(df[column])

    def build_frame(self, columns, index_names=(), index=True):
        return dict(columns)


def test_a_custom_adapter_inherits_the_numpy_path():
    adapter = _DictAdapter()
    df = {"a": np.array(["b", "a", "b"], dtype=object)}
    assert adapter._native_factorize(df, "a") is None
    assert_same_factorization(adapter.factorize(df, "a"), _reshape.factorize(df["a"]))


@settings(deadline=None, max_examples=200)
@given(labels=st.lists(st.text(max_size=5), min_size=1, max_size=30))
@pytest.mark.parametrize("engine_name", ["pandas", "polars"])
def test_factorize_always_matches_numpy(engine_name, labels):
    df = build_frame(engine_name, {"a": labels})
    assert_same_factorization(
        get_adapter(engine_name).factorize(df, "a"), reference(engine_name, df)
    )
