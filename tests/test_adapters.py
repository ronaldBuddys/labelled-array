"""Tests common to every DataFrame adapter, and the adapter registry."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import build_frame, column, columns, n_rows
from labelled_array import (
    AdapterError,
    DimensionError,
    DuplicateIndexError,
    LabelledArray,
)
from labelled_array.adapters import (
    DEFAULT_ADAPTERS,
    DataFrameAdapter,
    PandasAdapter,
    PolarsAdapter,
    available_adapters,
    get_adapter,
    register_adapter,
    registered_adapters,
    resolve_adapter,
)

# -- registry -------------------------------------------------------------------


def test_both_engines_are_registered_and_installed():
    assert set(registered_adapters()) == {"pandas", "polars"}
    assert set(available_adapters()) == {"pandas", "polars"}
    assert DEFAULT_ADAPTERS == ("polars", "pandas")


def test_get_adapter_defaults_to_polars():
    assert get_adapter().name == "polars"


def test_default_engine_falls_back_to_pandas_without_polars(monkeypatch, simple):
    # The install check is cached, so it is the adapter that has to say no.
    monkeypatch.setattr(PolarsAdapter, "is_available", classmethod(lambda cls: False))
    assert get_adapter().name == "pandas"
    assert type(simple.to_dataframe()).__module__.startswith("pandas")


def test_no_installed_engine_names_the_candidates(monkeypatch):
    for adapter in (PandasAdapter, PolarsAdapter):
        monkeypatch.setattr(adapter, "is_available", classmethod(lambda cls: False))
    with pytest.raises(AdapterError, match="none of the default DataFrame engines"):
        get_adapter()


def test_naming_an_engine_never_falls_back(monkeypatch):
    monkeypatch.setattr(PolarsAdapter, "is_available", classmethod(lambda cls: False))
    with pytest.raises(AdapterError, match="not installed"):
        get_adapter("polars")


def test_get_adapter_passes_through_instances():
    adapter = get_adapter("polars")
    assert get_adapter(adapter) is adapter


def test_get_unknown_adapter_raises():
    with pytest.raises(AdapterError, match="unknown DataFrame engine 'nope'"):
        get_adapter("nope")


def test_register_rejects_non_adapter():
    with pytest.raises(TypeError, match="expected a DataFrameAdapter"):
        register_adapter(object())


def test_registering_a_custom_adapter_needs_no_core_change():
    class DummyAdapter(PandasAdapter):
        name = "dummy"

    try:
        register_adapter(DummyAdapter)
        assert get_adapter("dummy").name == "dummy"
    finally:
        from labelled_array.adapters import _REGISTRY

        _REGISTRY.unregister("dummy")


def test_resolve_adapter_identifies_frames(engine, sales_data):
    assert resolve_adapter(build_frame(engine, sales_data)).name == engine


def test_resolve_unknown_object_raises():
    with pytest.raises(AdapterError, match="no registered DataFrame engine handles"):
        resolve_adapter({"a": [1]})


def test_adapter_equality_and_repr():
    assert get_adapter("pandas") == PandasAdapter()
    assert get_adapter("pandas") != get_adapter("polars")
    assert len({get_adapter("pandas"), PandasAdapter()}) == 1
    assert repr(PandasAdapter()) == "PandasAdapter()"


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        DataFrameAdapter()


# -- adapter methods ------------------------------------------------------------


def test_column_names_and_row_count(engine, sales_data):
    adapter = get_adapter(engine)
    df = build_frame(engine, sales_data)
    assert adapter.column_names(df) == list(sales_data)
    assert adapter.n_rows(df) == 4


def test_column_values_are_numpy(engine, sales_data):
    values = get_adapter(engine).column_values(build_frame(engine, sales_data), "units")
    assert isinstance(values, np.ndarray)
    assert values.tolist() == [10, 20, 30, 40]


def test_build_frame_round_trips_columns(engine):
    built = get_adapter(engine).build_frame(
        {"a": np.array([1, 2]), "w": np.array([1.5, 2.5])},
        index_names=["a"],
        index=False,
    )
    assert columns(built) == ["a", "w"]
    assert column(built, "w").tolist() == [1.5, 2.5]


# -- from_dataframe -------------------------------------------------------------


def test_from_dataframe_builds_a_dense_array(engine, sales_data):
    df = build_frame(engine, sales_data)
    la = LabelledArray.from_dataframe(df, dims=["region", "product"])
    assert la.dims.names == ("region", "product")
    assert la.shape == (2, 3)
    assert la.variables == ("units", "price")
    assert la.dims["region"].tolist() == ["north", "south"]
    assert la.dims["product"].tolist() == ["AAA", "BBB", "CCC"]
    absent = np.iinfo("int64").min  # unmistakable, where a zero would pass for data
    assert la["units"].tolist() == [[10, 20, 40], [30, absent, absent]]
    assert la.mask.tolist() == [[True, True, True], [True, False, False]]
    assert la.density == pytest.approx(4 / 6)


def test_labels_come_from_the_values_present_not_from_stale_categories(engine):
    """Regression: a filtered frame must not be padded out to unused labels."""
    data = {"a": [1, 2, 3], "b": ["x", "y", "z"], "w": [1.0, 2.0, 3.0]}
    filtered = {name: values[:1] for name, values in data.items()}
    la = LabelledArray.from_dataframe(build_frame(engine, filtered), dims=["a", "b"])
    assert la.shape == (1, 1)
    assert la.mask is None


def test_cols_default_to_every_non_dimension_column(engine, sales_data):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data), dims=["region", "product"]
    )
    assert la.variables == ("units", "price")


def test_cols_and_dims_accept_a_single_name(engine):
    data = {"region": ["north", "south"], "units": [10, 20], "price": [1.5, 2.5]}
    la = LabelledArray.from_dataframe(
        build_frame(engine, data), cols="units", dims="region"
    )
    assert la.variables == ("units",)
    assert la.dims.names == ("region",)


def test_cols_order_is_respected(engine, sales_data):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data),
        cols=["price", "units"],
        dims=["region", "product"],
    )
    assert la.variables == ("price", "units")


def test_dims_order_sets_the_axis_order(engine, sales_data):
    df = build_frame(engine, sales_data)
    la = LabelledArray.from_dataframe(df, dims=["product", "region"])
    assert la.shape == (3, 2)
    transposed = LabelledArray.from_dataframe(
        df, dims=["region", "product"]
    ).transpose()
    assert la.equals(transposed)


def test_unknown_columns_raise(engine, sales_data):
    df = build_frame(engine, sales_data)
    with pytest.raises(KeyError, match="columns not in the frame"):
        LabelledArray.from_dataframe(df, cols=["zz"], dims=["region"])
    with pytest.raises(KeyError, match="columns not in the frame"):
        LabelledArray.from_dataframe(df, dims=["zz"])


def test_a_column_cannot_be_both_dimension_and_variable(engine, sales_data):
    with pytest.raises(DimensionError, match="both dimensions and variables"):
        LabelledArray.from_dataframe(
            build_frame(engine, sales_data), cols=["region"], dims=["region"]
        )


def test_duplicate_rows_raise_with_a_useful_message(engine):
    data = {"a": [1, 1], "b": ["x", "x"], "w": [1.0, 2.0]}
    with pytest.raises(DuplicateIndexError) as excinfo:
        LabelledArray.from_dataframe(build_frame(engine, data), dims=["a", "b"])
    message = str(excinfo.value)
    assert "1 cell(s) are populated by more than one row" in message
    assert "['a', 'b']" in message
    assert "{'a': 1, 'b': 'x'}" in message, "the message names the offending labels"


def test_more_rows_than_a_zero_dimensional_array_can_hold(engine, sales_data):
    with pytest.raises(DuplicateIndexError, match="zero-dimensional"):
        LabelledArray.from_dataframe(
            build_frame(engine, sales_data), cols=["units"], dims=[]
        )


def test_fill_value_scalar_and_per_column(engine, sales_data):
    df = build_frame(engine, sales_data)
    la = LabelledArray.from_dataframe(df, dims=["region", "product"], fill_value=-1)
    assert la["units"].tolist()[1] == [30, -1, -1]
    assert la["price"].tolist()[1] == [3.5, -1.0, -1.0]

    la = LabelledArray.from_dataframe(
        df, dims=["region", "product"], fill_value={"units": -9}
    )
    assert la["units"][1, 1] == -9
    assert np.isnan(la["price"][1, 1]), "columns not named keep the default fill"


def test_dtype_scalar_and_per_column(engine, sales_data):
    df = build_frame(engine, sales_data)
    assert LabelledArray.from_dataframe(
        df, dims=["region", "product"], dtype="float32"
    ).dtypes == {"units": np.dtype("float32"), "price": np.dtype("float32")}
    la = LabelledArray.from_dataframe(
        df, dims=["region", "product"], dtype={"units": "float64"}
    )
    assert la.dtypes["units"] == np.dtype("float64")
    assert np.isnan(la["units"][1, 1]), "a float dtype gets a NaN fill"


def test_fill_value_and_dtype_reject_unknown_columns(engine, sales_data):
    df = build_frame(engine, sales_data)
    with pytest.raises(KeyError, match="fill_value given for unknown columns"):
        LabelledArray.from_dataframe(df, dims=["region"], fill_value={"zz": 0})
    with pytest.raises(KeyError, match="dtype given for unknown columns"):
        LabelledArray.from_dataframe(df, dims=["region"], dtype={"zz": float})


def test_mask_false_omits_the_mask(engine, sales_data):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data), dims=["region", "product"], mask=False
    )
    assert la.mask is None
    assert la.density == 1.0


def test_mask_is_dropped_when_every_cell_is_populated(engine):
    data = {"a": [1, 1, 2, 2], "b": ["x", "y", "x", "y"], "w": [1.0, 2.0, 3.0, 4.0]}
    assert (
        LabelledArray.from_dataframe(build_frame(engine, data), dims=["a", "b"]).mask
        is None
    )


def test_attrs_and_backend_arguments(engine, sales_data, backend):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data),
        cols=["price"],
        dims=["region", "product"],
        attrs={"source": "sales"},
        backend=backend,
    )
    assert la.attrs == {"source": "sales"}
    assert la.backend.name == backend.name


def test_engine_argument_forces_an_adapter(engine):
    data = {"region": ["north", "south"], "units": [10, 20]}
    la = LabelledArray.from_dataframe(
        build_frame(engine, data), dims=["region"], cols=["units"], engine=engine
    )
    assert la.shape == (2,)


def test_from_dataframe_returns_the_subclass(engine, sales_data):
    class Sales(LabelledArray):
        pass

    la = Sales.from_dataframe(
        build_frame(engine, sales_data), dims=["region", "product"]
    )
    assert type(la) is Sales


def test_single_dimension(engine):
    data = {"a": [3, 1, 2], "w": [3.0, 1.0, 2.0]}
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=["a"])
    assert la.shape == (3,)
    assert la.dims["a"].tolist() == [1, 2, 3], "labels are sorted"
    assert la["w"].tolist() == [1.0, 2.0, 3.0]


def test_three_dimensions(engine):
    data = {
        "a": [1, 1, 2, 2],
        "b": ["x", "x", "y", "y"],
        "c": [True, False, True, False],
        "w": [1.0, 2.0, 3.0, 4.0],
    }
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=["a", "b", "c"])
    assert la.shape == (2, 2, 2)
    assert la.density == 0.5


def test_empty_frame(engine):
    data = {
        "a": np.array([], dtype=np.int64),
        "b": np.array([], dtype=np.float64),
        "w": np.array([], dtype=np.float64),
    }
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=["a", "b"])
    assert la.shape == (0, 0)
    assert la.size == 0
    assert la.mask is None


def test_single_row(engine):
    la = LabelledArray.from_dataframe(
        build_frame(engine, {"a": [1], "b": ["x"], "w": [1.5]}), dims=["a", "b"]
    )
    assert la.shape == (1, 1)
    assert la["w"].tolist() == [[1.5]]


def test_zero_dimensional(engine):
    la = LabelledArray.from_dataframe(
        build_frame(engine, {"w": [1.5]}), cols=["w"], dims=[]
    )
    assert la.shape == ()
    assert la["w"].item() == 1.5


# -- to_dataframe ---------------------------------------------------------------


def test_to_dataframe_emits_one_row_per_populated_cell(engine, sales_data):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data), dims=["region", "product"]
    )
    out = la.to_dataframe(engine, index=False)
    assert n_rows(engine, out) == 4
    assert columns(out) == ["region", "product", "units", "price"]


def test_to_dataframe_without_a_mask_emits_every_cell(engine, sales_data):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data), dims=["region", "product"], mask=False
    )
    assert n_rows(engine, la.to_dataframe(engine, index=False)) == 6


def test_to_dataframe_dropna_removes_all_null_rows(engine, sales_data):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data),
        dims=["region", "product"],
        dtype="float64",
        mask=False,
    )
    out = la.to_dataframe(engine, index=False, dropna=True)
    assert n_rows(engine, out) == 4


def test_to_dataframe_dropna_keeps_partially_populated_rows(engine):
    data = {"a": [1, 2], "w": [1.0, np.nan], "v": [1.0, 2.0]}
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=["a"])
    assert n_rows(engine, la.to_dataframe(engine, index=False, dropna=True)) == 2


def test_to_dataframe_rows_are_ordered_by_the_dimension_labels(engine):
    data = {"a": [2, 1, 2, 1], "b": ["y", "y", "x", "x"], "w": [1.0, 2.0, 3.0, 4.0]}
    la = LabelledArray.from_dataframe(build_frame(engine, data), dims=["a", "b"])
    out = la.to_dataframe(engine, index=False)
    assert column(out, "a").tolist() == [1, 1, 2, 2]
    assert column(out, "b").tolist() == ["x", "y", "x", "y"]


def test_to_dataframe_of_an_empty_selection(engine, sales_data):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data), dims=["region", "product"]
    )
    assert n_rows(engine, la.isel(region=[]).to_dataframe(engine, index=False)) == 0


def test_to_dataframe_from_other_backends(engine, sales_data, backend):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data), cols=["price"], dims=["region", "product"]
    ).to_backend(backend)
    out = la.to_dataframe(engine, index=False)
    assert column(out, "price").tolist() == pytest.approx([1.5, 2.5, 4.5, 3.5])


def test_to_dataframe_of_a_zero_dimensional_array(engine):
    la = LabelledArray({"w": np.array(1.5)}, {})
    out = la.to_dataframe(engine, index=False)
    assert n_rows(engine, out) == 1
    assert columns(out) == ["w"]


def test_to_dataframe_with_no_variables(engine, sales_data):
    la = LabelledArray.from_dataframe(
        build_frame(engine, sales_data), dims=["region", "product"]
    ).drop_vars(["units", "price"])
    out = la.to_dataframe(engine, index=False)
    assert columns(out) == ["region", "product"]
