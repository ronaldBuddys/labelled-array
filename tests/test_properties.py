"""Property-based round trips over randomly generated frames."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from helpers import (
    assert_frames_equal,
    build_both,
    build_frame,
    needs_polars,
    sort_frame,
)
from labelled_array import LabelledArray

SETTINGS = settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

#: Label strategies, one per dtype we let dimensions take.
LABELS = {
    "int": st.integers(min_value=-1000, max_value=1000),
    "str": st.text(alphabet="abcdefgxyz", min_size=1, max_size=4),
    "float": st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
}

#: Value column dtypes, all of which must survive a sparse round trip exactly.
VALUE_DTYPES = ["int64", "int32", "float64", "float32", "bool"]


@st.composite
def frames(draw, max_dims=3, max_variables=2):
    """Generate a frame with unique dimension rows, plus its dimension names."""
    n_dims = draw(st.integers(min_value=1, max_value=max_dims))
    dim_names = [f"d{i}" for i in range(n_dims)]

    label_sets = [
        draw(
            st.lists(
                LABELS[draw(st.sampled_from(sorted(LABELS)))],
                min_size=1,
                max_size=4,
                unique=True,
            )
        )
        for _ in dim_names
    ]

    cells = [()]
    for labels in label_sets:
        cells = [cell + (label,) for cell in cells for label in labels]
    rows = draw(
        st.lists(st.sampled_from(cells), min_size=1, max_size=len(cells), unique=True)
    )

    data = {name: [row[axis] for row in rows] for axis, name in enumerate(dim_names)}
    n_variables = draw(st.integers(min_value=1, max_value=max_variables))
    for i in range(n_variables):
        dtype = np.dtype(draw(st.sampled_from(VALUE_DTYPES)))
        values = draw(
            st.lists(
                st.booleans()
                if dtype.kind == "b"
                else st.integers(min_value=-100, max_value=100),
                min_size=len(rows),
                max_size=len(rows),
            )
        )
        data[f"v{i}"] = np.array(values, dtype=dtype)
    return data, dim_names


@SETTINGS
@given(frame=frames())
def test_round_trip_is_the_identity(engine, frame):
    data, dim_names = frame
    df = build_frame(engine, data)
    la = LabelledArray.from_dataframe(df, dims=dim_names)
    out = la.to_dataframe(engine, index=False)
    assert_frames_equal(engine, sort_frame(engine, df, dim_names), out)


@needs_polars
@SETTINGS
@given(frame=frames())
def test_the_engines_always_agree(frame):
    data, dim_names = frame
    built = build_both(data, dim_names)
    assert built[0].equals(built[1])


@SETTINGS
@given(frame=frames())
def test_the_mask_counts_the_rows(engine, frame):
    data, dim_names = frame
    df = build_frame(engine, data)
    la = LabelledArray.from_dataframe(df, dims=dim_names)
    populated = la.size if la.mask is None else int(la.mask.sum())
    assert populated == len(data[dim_names[0]])
    assert la.density == pytest.approx(populated / la.size)


@SETTINGS
@given(frame=frames())
def test_dtypes_are_preserved(engine, frame):
    data, dim_names = frame
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=dim_names)
    for name, values in data.items():
        if name not in dim_names:
            assert la.dtypes[name] == values.dtype


@SETTINGS
@given(frame=frames())
def test_selecting_every_label_is_a_no_op(engine, frame):
    data, dim_names = frame
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=dim_names)
    everything = {name: la.dims[name] for name in dim_names}
    assert la.select(everything).equals(la)


@SETTINGS
@given(frame=frames())
def test_a_selection_holds_the_same_values_as_the_source(engine, frame):
    data, dim_names = frame
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=dim_names)
    name = dim_names[0]
    wanted = la.dims[name][:1]
    selected = la.select({name: wanted})
    assert selected.dims[name].tolist() == wanted.tolist()
    for variable in la.variables:
        np.testing.assert_array_equal(selected[variable], la[variable][:1])


@SETTINGS
@given(frame=frames())
def test_transposing_twice_is_a_no_op(engine, frame):
    data, dim_names = frame
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=dim_names)
    assert la.transpose().transpose().equals(la)


@SETTINGS
@given(frame=frames())
def test_repr_round_trips(engine, frame):
    from numpy import array  # noqa: F401

    from labelled_array import Dimensions  # noqa: F401

    data, dim_names = frame
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=dim_names)
    assert eval(la.to_repr(full=True)) == la
