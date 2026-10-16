"""pandas and polars frames are interchangeable on the public API.

``test_parity`` is the smoke test; this is the matrix. Every ``from_dataframe``
option, label dtype and frame shape is built from both libraries, and through
both the native and the numpy factorization, and has to give the same array.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from helpers import build_frame, column, needs_polars
from labelled_array import (
    DimensionError,
    DuplicateIndexError,
    LabelledArray,
)
from labelled_array._utils import arrays_equal
from labelled_array.adapters import PolarsAdapter
from labelled_array.backends import available_backends

ENGINES = ("pandas", "polars")

#: Frames of every shape the conversion distinguishes, with their dimensions.
DATASETS = {
    "sparse": (
        {
            "region": ["north", "north", "south", "north"],
            "product": ["AAA", "BBB", "AAA", "CCC"],
            "units": [10, 20, 30, 40],
            "price": [1.5, 2.5, 3.5, 4.5],
        },
        ["region", "product"],
    ),
    "dense": (
        {
            "region": ["north", "north", "south", "south"],
            "product": ["AAA", "BBB", "AAA", "BBB"],
            "units": [1, 2, 3, 4],
        },
        ["region", "product"],
    ),
    "unsorted_rows": (
        {
            "region": ["south", "north", "south", "north"],
            "product": ["BBB", "CCC", "AAA", "AAA"],
            "units": [4, 3, 2, 1],
        },
        ["region", "product"],
    ),
    "unicode_labels": (
        {
            "region": ["\u00e9", "e\u0301", "z", "", "\U0001f600"],
            "product": ["B", "a", "\u00df", "ss", " "],
            "units": [1, 2, 3, 4, 5],
        },
        ["region", "product"],
    ),
    "single_row": ({"region": ["north"], "units": [1]}, ["region"]),
    "zero_rows": (
        {
            "region": np.array([], dtype="int64"),
            "units": np.array([], dtype="float64"),
        },
        ["region"],
    ),
    "one_dimension": ({"region": ["s", "n", "e"], "units": [1, 2, 3]}, ["region"]),
    "three_dimensions": (
        {
            "a": ["x", "x", "y", "y"],
            "b": [2, 1, 2, 1],
            "c": [True, False, False, True],
            "units": [1.0, 2.0, 3.0, 4.0],
        },
        ["a", "b", "c"],
    ),
}

#: One dimension's labels per dtype family a dimension can take.
LABEL_FAMILIES = {
    "str": ["b", "a", "c"],
    "int": np.array([3, 1, 2], dtype="int64"),
    "float": np.array([2.5, 0.5, 1.5]),
    "bool": np.array([True, False, False]),
    "datetime": np.array(["2024-03-01", "2024-01-01", "2024-02-01"], "datetime64[us]"),
    "timedelta": np.array([3, 1, 2], dtype="timedelta64[ms]"),
}

#: A frame with a descriptive column and a date, for the options that need one.
BOOK = {
    "asset": ["BBB", "AAA", "AAA", "CCC", "BBB"],
    "date": np.array(
        ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"],
        dtype="datetime64[us]",
    ),
    "sector": ["tech", "fin", "fin", "tech", "tech"],
    "px": [20.0, 10.0, 11.0, 30.0, 21.0],
    "qty": [2, 1, 1, 3, 2],
}
BOOK_DIMS = ["asset", "date"]

#: Every ``from_dataframe`` option, as keyword arguments over ``BOOK``.
KWARGS = {
    "defaults": {},
    "cols_as_a_string": {"cols": "px"},
    "cols_as_a_subset": {"cols": ["qty", "px"]},
    "dims_reversed": {"dims": ["date", "asset"]},
    "natural": {"cols": ["px", "qty"], "natural": ["date"]},
    "alt": {"cols": ["px"], "alt": {"sector": "asset"}},
    "fill_value_scalar": {"cols": ["px"], "fill_value": -1.0},
    "fill_value_mapping": {"cols": ["px", "qty"], "fill_value": {"px": 0.0, "qty": -1}},
    "dtype_scalar": {"cols": ["px", "qty"], "dtype": "float32"},
    "dtype_mapping": {
        "cols": ["px", "qty"],
        "dtype": {"px": "float32", "qty": "int32"},
    },
    "no_mask": {"mask": False},
    "attrs": {"attrs": {"source": "test"}},
}
KWARGS.update(
    {
        f"backend_{name}": {"cols": ["px"], "backend": name}
        for name in available_backends()
    }
)


def build(library: str, data, dims, /, **kwargs) -> LabelledArray:
    """Build the array from ``library``'s frame of ``data``; ``kwargs`` win."""
    kwargs = {"dims": dims, **kwargs}
    return LabelledArray.from_dataframe(build_frame(library, data), **kwargs)


def assert_same_array(left: LabelledArray, right: LabelledArray) -> None:
    """Assert two arrays agree, including the dtypes ``equals`` lets differ."""
    assert left.equals(right)
    assert left.dims.names == right.dims.names
    for name in left.dims.names:
        assert left.dims[name].dtype == right.dims[name].dtype, name
    assert left.dtypes == right.dtypes
    assert (left.mask is None) == (right.mask is None)
    if left.mask is not None:
        np.testing.assert_array_equal(left.mask, right.mask)
    assert list(left.coords) == list(right.coords)
    for name in left.coords:
        assert left.coords[name].dtype == right.coords[name].dtype, name
    assert list(left.dims.alt) == list(right.dims.alt)
    for name in left.dims.alt:
        assert left.dims.alt[name].dtype == right.dims.alt[name].dtype, name
        assert arrays_equal(left.dims.alt[name], right.dims.alt[name])
    assert left.attrs == right.attrs
    assert left.backend == right.backend


@pytest.fixture
def numpy_only(monkeypatch):
    """Switch the native factorization off, leaving the numpy reference."""
    monkeypatch.setattr(
        PolarsAdapter, "_native_factorize", lambda self, df, column: None
    )


# -- the two engines ---------------------------------------------------------


@needs_polars
@pytest.mark.parametrize("name", sorted(DATASETS))
def test_engines_build_the_same_array_from_every_shape(name):
    data, dims = DATASETS[name]
    assert_same_array(*(build(engine, data, dims) for engine in ENGINES))


@needs_polars
@pytest.mark.parametrize("family", sorted(LABEL_FAMILIES))
def test_engines_build_the_same_array_from_every_label_dtype(family):
    data = {"label": LABEL_FAMILIES[family], "other": ["x", "y", "x"], "w": [1, 2, 3]}
    assert_same_array(*(build(engine, data, ["label", "other"]) for engine in ENGINES))


@needs_polars
@pytest.mark.parametrize("name", sorted(KWARGS))
def test_engines_build_the_same_array_with_every_option(name):
    assert_same_array(
        *(build(engine, BOOK, BOOK_DIMS, **KWARGS[name]) for engine in ENGINES)
    )


def test_forcing_the_engine_changes_nothing(engine):
    detected = build(engine, BOOK, BOOK_DIMS)
    forced = build(engine, BOOK, BOOK_DIMS, engine=engine)
    assert_same_array(detected, forced)


@needs_polars
def test_date_labels_are_the_documented_exception():
    # polars has a Date dtype and gives back datetime64[D]; pandas has none and
    # keeps the python objects. The labels agree, their dtype does not.
    days = [datetime.date(2024, 1, 2), datetime.date(2024, 1, 1)]
    data = {"day": days, "w": [1.0, 2.0]}
    from_pandas, from_polars = (build(engine, data, ["day"]) for engine in ENGINES)
    assert from_pandas.dims["day"].dtype == np.dtype("O")
    assert from_polars.dims["day"].dtype == np.dtype("datetime64[D]")
    assert from_pandas.dims["day"].tolist() == from_polars.dims["day"].tolist()
    assert from_pandas["w"].tolist() == from_polars["w"].tolist()


# -- the two factorizations --------------------------------------------------


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_native_and_numpy_paths_agree_on_every_shape(engine, name, request):
    data, dims = DATASETS[name]
    native = build(engine, data, dims)
    request.getfixturevalue("numpy_only")
    assert_same_array(native, build(engine, data, dims))


@pytest.mark.parametrize("name", sorted(KWARGS))
def test_native_and_numpy_paths_agree_with_every_option(engine, name, request):
    native = build(engine, BOOK, BOOK_DIMS, **KWARGS[name])
    request.getfixturevalue("numpy_only")
    assert_same_array(native, build(engine, BOOK, BOOK_DIMS, **KWARGS[name]))


# -- writing out -------------------------------------------------------------


def assert_same_content(from_pandas, from_polars) -> None:
    """Assert a pandas and a polars frame hold the same columns and values."""
    assert list(from_pandas.columns) == list(from_polars.columns)
    for name in from_polars.columns:
        assert arrays_equal(column(from_pandas, name), column(from_polars, name)), name


@needs_polars
@pytest.mark.parametrize("name", sorted(DATASETS))
def test_engines_write_the_same_content(engine, name):
    data, dims = DATASETS[name]
    la = build(engine, data, dims)
    assert_same_content(
        la.to_dataframe("pandas", index=False), la.to_dataframe("polars")
    )


@needs_polars
@pytest.mark.parametrize("name", ["natural", "alt", "no_mask", "dims_reversed"])
def test_engines_write_the_same_content_with_options(engine, name):
    la = build(engine, BOOK, BOOK_DIMS, **KWARGS[name])
    assert_same_content(
        la.to_dataframe("pandas", index=False), la.to_dataframe("polars")
    )


@needs_polars
@pytest.mark.parametrize("dropna", [False, True])
@pytest.mark.parametrize("expand_natural", [False, True])
def test_engines_write_the_same_content_with_every_flag(dropna, expand_natural):
    la = build("polars", BOOK, BOOK_DIMS, cols=["px"], natural=["date"], mask=False)
    flags = {"dropna": dropna, "expand_natural": expand_natural}
    assert_same_content(
        la.to_dataframe("pandas", index=False, **flags),
        la.to_dataframe("polars", **flags),
    )


@needs_polars
@pytest.mark.parametrize("name", sorted(set(DATASETS) - {"zero_rows"}))
@pytest.mark.parametrize("source, via", [ENGINES, ENGINES[::-1]])
def test_cross_engine_round_trips(name, source, via):
    data, dims = DATASETS[name]
    la = build(source, data, dims)
    written = la.to_dataframe(via, index=False)
    assert_same_array(la, LabelledArray.from_dataframe(written, dims=dims))


# -- refusing ----------------------------------------------------------------

REFUSED = {
    "duplicate_cell": (
        {"a": ["x", "x"], "w": [1, 2]},
        {"dims": ["a"]},
        DuplicateIndexError,
    ),
    "null_label": ({"a": ["x", None], "w": [1, 2]}, {"dims": ["a"]}, DimensionError),
    "natural_with_repeats": (
        {"a": ["x", "x"], "d": [1, 1], "w": [1, 2]},
        {"dims": ["a", "d"], "natural": ["d"]},
        DuplicateIndexError,
    ),
    "alt_not_constant": (
        {"a": ["x", "x"], "d": [1, 2], "s": ["p", "q"], "w": [1, 2]},
        {"dims": ["a", "d"], "cols": ["w"], "alt": {"s": "a"}},
        DimensionError,
    ),
    "missing_column": ({"a": ["x"], "w": [1]}, {"dims": ["a", "nope"]}, KeyError),
    "no_dims": ({"a": ["x"], "w": [1]}, {}, DimensionError),
}


@needs_polars
@pytest.mark.parametrize("name", sorted(REFUSED))
def test_engines_refuse_the_same_inputs_with_the_same_message(name):
    data, kwargs, error = REFUSED[name]
    messages = []
    for engine in ENGINES:
        with pytest.raises(error) as raised:
            LabelledArray.from_dataframe(build_frame(engine, data), **kwargs)
        messages.append(str(raised.value))
    assert messages[0] == messages[1]


# -- any strings at all ------------------------------------------------------


@st.composite
def string_frames(draw):
    """Generate a sparse two-dimensional frame labelled by arbitrary text."""
    text = st.text(max_size=4)
    rows = draw(st.lists(st.tuples(text, text), min_size=1, max_size=12, unique=True))
    return {
        "a": [row[0] for row in rows],
        "b": [row[1] for row in rows],
        "w": list(range(len(rows))),
    }


@needs_polars
@settings(
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(data=string_frames())
def test_engines_always_agree_on_arbitrary_text_labels(data):
    assert_same_array(*(build(engine, data, ["a", "b"]) for engine in ENGINES))
